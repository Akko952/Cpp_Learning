#ifndef NEURALNETWORKSOFTMAX_HPP
#define NEURALNETWORKSOFTMAX_HPP

#include "Layer.hpp"
#include "Matrix.hpp"
#include "SoftmaxCrossEntropy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

enum class InitMethod { Custom, Normal, Xavier, He };

class NeuralNetworkSoftmax {
  private:
    std::vector<Layer> layers;
    std::vector<Matrix> weights;

    void connect_layers() {
        if (layers.size() < 2) {
            throw std::runtime_error(
                "NeuralNetworkSoftmax must contain at least input and output layers.");
        }

        for (size_t i = 0; i < layers.size() - 1; ++i) {
            weights.emplace_back(layers[i].lay_size, layers[i + 1].lay_size);
        }
    }

    void initialize_weights(InitMethod method = InitMethod::Custom) {
        static std::mt19937 gen(std::random_device{}());
        for (size_t idx = 0; idx < weights.size(); ++idx) {
            auto &weight_matrix = weights[idx];
            size_t fan_in  = weight_matrix.getRows();
            size_t fan_out = weight_matrix.getCols();

            if (method == InitMethod::Custom) {
                weight_matrix.fillRandom(-1.0, 1.0);
                weight_matrix *= (2.0 / fan_in);
            } else {
                double std_dev = 0.01;
                if (method == InitMethod::Xavier) {
                    std_dev = std::sqrt(2.0 / (fan_in + fan_out));
                } else if (method == InitMethod::He) {
                    std_dev = std::sqrt(2.0 / fan_in);
                }
                std::normal_distribution<double> dist(0.0, std_dev);
                for (size_t i = 0; i < fan_in; ++i) {
                    for (size_t j = 0; j < fan_out; ++j) {
                        weight_matrix(i, j) = dist(gen);
                    }
                }
            }
        }
    }

  public:
    NeuralNetworkSoftmax(const std::vector<Layer> &network_layer,
                         InitMethod init_method = InitMethod::Custom)
        : layers(network_layer) {
        connect_layers();
        initialize_weights(init_method);
    }

    const std::vector<Layer> &getLayers() const {
        return layers;
    }

    std::vector<double> forward(const std::vector<double> &input) {
        if (input.size() != layers[0].lay_size) {
            throw std::runtime_error(
                "Input size does not match the first layer size.");
        }

        layers[0].A_output = input;

        for (size_t layerIndex = 1; layerIndex < layers.size(); ++layerIndex) {
            Matrix &weight_matrix = weights[layerIndex - 1];
            Layer &current_layer = layers[layerIndex];
            Layer &previous_layer = layers[layerIndex - 1];

            for (size_t j = 0; j < current_layer.lay_size; ++j) {
                double sum = current_layer.bias[j];
                for (size_t i = 0; i < previous_layer.lay_size; ++i) {
                    sum += previous_layer.A_output[i] * weight_matrix(i, j);
                }
                current_layer.Z_input[j] = sum;

                if (layerIndex == layers.size() - 1) {
                    current_layer.A_output[j] = sum;
                } else {
                    current_layer.A_output[j] =
                        current_layer.applyActivationFunction(sum);
                }
            }
        }

        Layer &output_layer = layers.back();
        output_layer.A_output = SoftmaxCrossEntropy::softmax(output_layer.Z_input);
        return output_layer.A_output;
    }

    std::vector<double> predict(const std::vector<double> &input) {
        return forward(input);
    }

    size_t predictClass(const std::vector<double> &input) {
        return SoftmaxCrossEntropy::predictClass(predict(input));
    }

    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets,
               double learning_rate, size_t epochs,
               size_t batch_size = 1, bool verbose = true,
               size_t batch_report_interval = 0,
               size_t start_epoch = 0,
               const std::string &save_prefix = "model") {
        if (inputs.size() != targets.size()) {
            throw std::runtime_error(
                "Inputs and targets must have the same number of samples.");
        }
        if (inputs.empty()) {
            throw std::runtime_error("Training inputs must not be empty.");
        }
        if (learning_rate <= 0.0 || epochs == 0 || batch_size == 0) {
            throw std::runtime_error(
                "Learning rate, epochs, and batch size must be positive.");
        }
        if (inputs[0].size() != layers[0].lay_size) {
            throw std::runtime_error(
                "Training input dimension does not match the network input layer.");
        }
        if (targets[0].size() != layers.back().lay_size) {
            throw std::runtime_error(
                "Training target dimension does not match the network output layer.");
        }

        const size_t dataset_size = inputs.size();
        const size_t total_batches = (dataset_size + batch_size - 1) / batch_size;
        const size_t effective_batch_report_interval =
            batch_report_interval == 0 ? std::max<size_t>(1, total_batches / 10)
                                       : batch_report_interval;
        const auto start = std::chrono::high_resolution_clock::now();
        double base_lr = learning_rate;
        double decay_rate = 0.996;
        double min_lr = 0.0001;

        if (verbose) {
            std::stringstream data;
            data << "Training started: " << epochs << " epochs (from epoch " << (start_epoch + 1) << ")"
                 << ", " << dataset_size << " samples, batch size: " << batch_size
                 << ", initial learning rate: " << learning_rate << "\n";
            data << "learning rate decay: " << (1 - decay_rate) * 100
                 << "%, minimum learning rate: " << min_lr << "\n";
            data << "------------------------------------------------------------\n";
            std::cout << data.str();
        }

        for (size_t epoch = start_epoch; epoch < epochs; ++epoch) {
            learning_rate = std::max(min_lr, base_lr * std::pow(decay_rate, epoch));
            double totalLoss = 0.0;

            for (size_t batch = 0; batch < dataset_size; batch += batch_size) {
                const size_t current_batch_size =
                    std::min(batch_size, dataset_size - batch);
                const size_t batch_index = batch / batch_size;

                if (verbose && (batch_index % effective_batch_report_interval == 0 ||
                                batch + current_batch_size >= dataset_size)) {
                    std::stringstream progress;
                    progress << "Epoch " << (epoch + 1) << "/" << epochs
                             << ", batch " << (batch_index + 1) << "/" << total_batches
                             << std::endl;
                    std::cout << progress.str();
                }

                std::vector<Matrix> weight_batch_gradients;
                weight_batch_gradients.reserve(weights.size());
                for (const auto &weight_matrix : weights) {
                    weight_batch_gradients.emplace_back(weight_matrix.getRows(),
                                                        weight_matrix.getCols());
                }

                std::vector<std::vector<double>> bias_batch_gradients(
                    layers.size() - 1);
                for (size_t i = 0; i < bias_batch_gradients.size(); ++i) {
                    bias_batch_gradients[i].resize(layers[i + 1].lay_size, 0.0);
                }

                for (size_t k = batch; k < batch + current_batch_size; ++k) {
                    forward(inputs[k]);

                    Layer &output_layer = layers.back();
                    totalLoss += SoftmaxCrossEntropy::softmaxCrossEntropyLoss(
                        output_layer.Z_input, targets[k]);

                    output_layer.gradient =
                        SoftmaxCrossEntropy::softmaxCrossEntropyGradient(
                            output_layer.Z_input, targets[k]);

                    for (int layerIndex = static_cast<int>(layers.size()) - 2;
                         layerIndex > 0; --layerIndex) {
                        for (size_t i = 0; i < layers[layerIndex].lay_size; ++i) {
                            double error = 0.0;
                            for (size_t j = 0; j < weights[layerIndex].getCols(); ++j) {
                                error += weights[layerIndex](i, j) *
                                         layers[layerIndex + 1].gradient[j];
                            }
                            layers[layerIndex].gradient[i] =
                                error * layers[layerIndex].applyActivationFunctionDerivative(
                                            layers[layerIndex].Z_input[i]);
                        }
                    }

                    for (size_t weightIndex = 0; weightIndex < weights.size(); ++weightIndex) {
                        const std::vector<double> &previousActivations =
                            layers[weightIndex].A_output;
                        const std::vector<double> &nextGradient =
                            layers[weightIndex + 1].gradient;

                        for (size_t i = 0; i < weight_batch_gradients[weightIndex].getRows(); ++i) {
                            for (size_t j = 0; j < weight_batch_gradients[weightIndex].getCols(); ++j) {
                                weight_batch_gradients[weightIndex](i, j) +=
                                    previousActivations[i] * nextGradient[j];
                            }
                        }

                        for (size_t i = 0; i < layers[weightIndex + 1].lay_size; ++i) {
                            bias_batch_gradients[weightIndex][i] += nextGradient[i];
                        }
                    }
                }

                for (size_t weightIndex = 0; weightIndex < weights.size(); ++weightIndex) {
                    for (size_t i = 0; i < weights[weightIndex].getRows(); ++i) {
                        for (size_t j = 0; j < weights[weightIndex].getCols(); ++j) {
                            weights[weightIndex](i, j) -=
                                (learning_rate / current_batch_size) *
                                weight_batch_gradients[weightIndex](i, j);
                        }
                    }

                    for (size_t i = 0; i < layers[weightIndex + 1].lay_size; ++i) {
                        layers[weightIndex + 1].bias[i] -=
                            (learning_rate / current_batch_size) *
                            bias_batch_gradients[weightIndex][i];
                    }
                }
            }

              if (verbose) {
                std::stringstream data;
                data << "[" << ((epoch + 1) * 100 / epochs) << "%] Epoch: "
                     << (epoch + 1)
                     << " | CE: " << totalLoss / inputs.size() << "\n";
                std::cout << data.str();
            }

            saveModel(save_prefix + "_epoch_" + std::to_string(epoch + 1) + ".csv");
        }

        const auto end = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::stringstream data;
            data << "Training completed in "
                 << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
                 << " seconds.\n";
            std::cout << data.str();
        }
    }

    void saveModel(const std::string &filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }
        file << "type,layer,row,col,value\n";
        for (size_t l = 1; l < layers.size(); ++l) {
            for (size_t i = 0; i < weights[l - 1].getRows(); ++i) {
                for (size_t j = 0; j < weights[l - 1].getCols(); ++j) {
                    file << "weight," << l << "," << i << "," << j << ","
                         << weights[l - 1](i, j) << "\n";
                }
            }
            for (size_t i = 0; i < layers[l].bias.size(); ++i) {
                file << "bias," << l << ",," << i << ","
                     << layers[l].bias[i] << "\n";
            }
        }
        file.close();
        std::cout << "Model saved to " << filename << std::endl;
    }

    void loadModel(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string type, layerStr, rowStr, colStr, valueStr;
            std::getline(ss, type, ',');
            std::getline(ss, layerStr, ',');
            std::getline(ss, rowStr, ',');
            std::getline(ss, colStr, ',');
            std::getline(ss, valueStr, ',');
            int layer = std::stoi(layerStr);
            double value = std::stod(valueStr);
            if (type == "weight") {
                weights[layer - 1](std::stoi(rowStr), std::stoi(colStr)) = value;
            } else if (type == "bias") {
                layers[layer].bias[std::stoi(colStr)] = value;
            }
        }
        file.close();
        std::cout << "Model loaded from " << filename << std::endl;
    }
};

#endif // NEURALNETWORKSOFTMAX_HPP
