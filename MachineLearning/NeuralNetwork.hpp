#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include "Matrix.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <sstream>
#include "log.hpp" 
#include <fstream>

//activating_Function是一个函数对象类型，表示激活函数，它接受一个double类型的输入并返回一个double类型的输出。
//通过使用std::function<double(double)>,将不同的激活函数（如ReLU、Sigmoid等）封装为函数对象
//namespace ActivationFunctions:自定义命名空间
//enum class ActivationFunctionType:定义激活函数类型枚举


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
	void train(const std::vector<std::vector<double>> &inputs,
		   const std::vector<std::vector<double>> &targets,
		   double learning_rate, size_t epochs,
		   size_t batch_size=1,bool verbose=true) {
			if(inputs.size() != targets.size()) {
				throw std::runtime_error("Inputs and targets must have the same number of samples.");
		   }		
		   if (learning_rate <= 0|| epochs <= 0 || batch_size <= 0) {	
				throw std::runtime_error("Learning rate, epochs, and batch size must be positive.");
		   }
		   
		   size_t dataset_size = inputs.size();
		   auto start=std::chrono::high_resolution_clock::now();
		   double totalError=0.0;
		   double base_lr=learning_rate;
		   double decay_rate=0.996; //学习率衰减率,约为0.4%每次迭代
		   double min_lr=0.0001; //最小学习率

		if(verbose){
			std::stringstream data;
			data <<"Training started: "<<epochs<<" epochs, "<<dataset_size<<" samples, batch size: "<<batch_size<<", initial learning rate: "<<learning_rate<<"\n";
			data <<"learing rate decay: "<<(1-decay_rate)*100<<"%, minimum learning rate: "<<min_lr<<"\n";
			data<<"------------------------------------------------------------\n";
			std::cout<<data.str();
			Logger::log(data.str());
		}

		for (int epoch =0; epoch < epochs; ++epoch) {
			learning_rate = std::max(min_lr,base_lr* std::pow(decay_rate, epoch));
			totalError=0.0;//重头开始

			for (size_t batch = 0; batch < dataset_size; batch += batch_size) {
				size_t current_batch_size = std::min(batch_size, dataset_size - batch);

				std::vector<Matrix> weight_batch_gradients;
				for (size_t i = 0; i < weights.size(); ++i) {
					weight_batch_gradients.emplace_back(weights[i].getRows(), weights[i].getCols());
					
				}
				std::vector<std::vector<double>> bias_batch_gradients(layers.size() - 1);
				for (size_t i = 0; i < bias_batch_gradients.size(); ++i) {
					bias_batch_gradients[i].resize(layers[i + 1].lay_size, 0.0);
				}

				for (size_t k =batch;k<(batch+current_batch_size);++k){
					forward(inputs[k]);

					Layer &output_layer = layers.back();

					const double epsilon = 1e-8;

					for (size_t i = 0; i < output_layer.lay_size; ++i) {
						double y_true = targets[k][i];
						double y_pred = output_layer.A_output[i];

						y_pred =std::min(std::max(y_pred, epsilon), 1.0 - epsilon);

						totalError += (y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred));//二元交叉熵
						
						output_layer.gradient[i] = y_pred - y_true;
					}

					for (int l=(static_cast<int>(layers.size())-2);l>0;--l){
						for(size_t i=0;i<weights[l].getRows();++i){
							double error=0.0;
							for(size_t j=0;j<weights[l].getCols();++j){
								error+=weights[l](i,j)*layers[l+1].gradient[j];
							}
						layers[l].gradient[i]=error*
							layers[l].applyActivationFunctionDerivative(layers[l].Z_input[i]);

						}

					}

					for(int l = (static_cast<int>(layers.size())-2); l > 0; --l){
						const std::vector<double>& ActivationFunctions=(l==0)?inputs[k]:layers[l].A_output;

						for(size_t i=0;i<weight_batch_gradients[l].getRows();++i){
							for(size_t j=0;j<weight_batch_gradients[l].getCols();++j){
								weight_batch_gradients[l](i,j)+=ActivationFunctions[i]*layers[l+1].gradient[j];
							}
						}

						for(size_t i=0;i<layers[l+1].lay_size;++i){
							bias_batch_gradients[l][i]+=layers[l+1].gradient[i];
						}
					}

				}
				for (int l=(static_cast<int>(layers.size())-2); l > 0; --l){
					for(size_t i=0;i<weights[l].getRows();++i){
						for(size_t j=0;j<weights[l].getCols();++j){
							weights[l](i,j)-=(learning_rate/current_batch_size)*weight_batch_gradients[l](i,j);
						}
					}
					for(size_t i=0;i<layers[l+1].lay_size;++i){
						layers[l+1].bias[i]-=(learning_rate/current_batch_size)*bias_batch_gradients[l][i];
					}

					for (size_t i=0;i<layers[l+1].lay_size;++i){
						layers[l+1].bias[i]-=(learning_rate/current_batch_size)*bias_batch_gradients[l][i];
					}
				}
			}

			if(verbose && epoch%100==0){
				std::stringstream data;

				data<<"["
				<<(epoch*100/epochs) <<"%] Epoch: "<<epoch
				<<"| BCE :"<<totalError/inputs.size()<<"\n";
				std::cout<<data.str();

				std::cout<<data.str();
				Logger::log(data.str());
			}
		}

		auto end =std::chrono::high_resolution_clock::now();

		if(verbose){
			std::stringstream data;
			data<<"Training completed in "
			<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count()
			<<" seconds.\n";
			std::cout<<data.str();
			Logger::log(data.str());
		}
	}

	void saveModel(const std::string &filename="model.csv")const
	 {
		std::ofstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Unable to open file for writing: " << filename << std::endl;
			return;
		}

		file << "type,layer,row,col,value\n";
		for (size_t l = 1; l < layers.size(); ++l) {
			for (size_t i = 0; i < weights[l-1].getRows(); ++i) {
				for (size_t j = 0; j < weights[l-1].getCols(); ++j) {
					file << "weight," << l << "," << i << "," << j << "," << weights[l-1](i, j) << "\n";
				}
			}
			for (size_t i = 0; i < layers[l].bias.size(); ++i) {
				file << "bias," << l << ",," << i << "," << layers[l].bias[i] << "\n";
			}
		}
		file.close();
		std::cout << "Model saved to " << filename << std::endl;
	}


	void loadModel(const std::string &filename="model.csv") {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Unable to open file for reading: " << filename << std::endl;
			return;
		}

		std::string line;
		std::getline(file, line); // Skip header	

		while(std::getline(file, line)) {
			std::stringstream ss(line);
			std::string type,layerStr,rowStr,colStr,valueStr;

			std::getline(ss, type, ',');
			std::getline(ss, layerStr, ',');
			std::getline(ss, rowStr, ',');	
			std::getline(ss, colStr, ',');
			std::getline(ss, valueStr, ',');

			int layer = std::stoi(layerStr);
			int row = std::stoi(rowStr);
			int col = std::stoi(colStr);
			double value = std::stod(valueStr);

			if(type=="weight"){
				weights[layer-1](row, col) = value;
			}
			else if(type=="bias"){
				layers[layer].bias[row] = value;
			}
		}

		file.close();
		std::cout << "Model loaded from " << filename << std::endl;
	}
};
#endif