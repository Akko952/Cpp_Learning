#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include "Matrix.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

class NeuralNetwork {
  private:
    std ::vector<Layer> layers;
    std ::vector<Matrix> weights; // 权重矩阵

    void connect_layers() {
	for (size_t i = 0; i < layers.size() - 1; i++) {
	    weights.emplace_back(layers[i].lay_size, layers[i + 1].lay_size);
	};
    }

    void initialize_weights() {
	for (auto &weight_matrix : weights) {
	    weight_matrix.fillRandom(-1.0, 1.0);
	    weight_matrix *= (2.0 / weight_matrix.getRows()); // 缩小权重范围
	}
    }

  public:
    NeuralNetwork(const std::vector<Layer> &network_layer)
	: layers(network_layer) {
	connect_layers();
	initialize_weights();
    }

    std ::vector<double> forward(const std::vector<double> &input) {
	if (input.size() != layers[0].lay_size) {
	    throw std::runtime_error(
		"Input size does not match the first layer size.");
	}

	layers[0].A_output = input;
	for (size_t l = 1; l < layers.size() - 1; ++l) {
	    for (size_t j = 0; j < weights[l].getCols(); ++j) {
		double sum = layers[l + 1].bias[j];

		for (size_t i = 0; i < weights[l].getRows(); ++i) {
		    sum += layers[l].A_output[i] * weights[l](i, j);
		}
		layers[l + 1].Z_input[j] = sum;
		layers[l + 1].A_output[j] =
		    layers[l + 1].applyActivationFunction(sum);
	    }
	}
	return layers.back().A_output;
    }

    std::vector<double> predict(const std::vector<double> &input) {
	return forward(input);
    }
};

#endif