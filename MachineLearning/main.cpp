#include "MNISTLoader.hpp"
#include "NeuralNetwork.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Loading Fashion-MNIST..." << std::endl;

    auto trainImages = readImages("data/train-images-idx3-ubyte");
    auto trainLabels = readLabels("data/train-labels-idx1-ubyte");
    auto testImages  = readImages("data/t10k-images-idx3-ubyte");
    auto testLabels  = readLabels("data/t10k-labels-idx1-ubyte");

    std::vector<std::vector<double>> trainInputs;
    std::vector<std::vector<double>> trainTargets;

    size_t trainCount = std::min<size_t>(1000, trainImages.size());
    trainInputs.reserve(trainCount);
    trainTargets.reserve(trainCount);

    for (size_t i = 0; i < trainCount; ++i) {
        trainInputs.push_back(trainImages[i]);
        trainTargets.push_back(labelToOneHot(trainLabels[i]));
    }

    std::cout << "Train: " << trainImages.size() << " images" << std::endl;
    std::cout << "Test:  " << testImages.size()  << " images" << std::endl;
    std::cout << "First image size: " << trainImages[0].size() << " pixels" << std::endl;
    std::cout << "First label: " << trainLabels[0] << std::endl;

    std::vector<Layer> layers;
    layers.emplace_back(0, 784, ActivationFunctionType::None);
    layers.emplace_back(1, 128, ActivationFunctionType::RELU);
    layers.emplace_back(2, 64, ActivationFunctionType::RELU);
    layers.emplace_back(3, 10, ActivationFunctionType::SIGMOID);

    NeuralNetwork network(layers);

    std::cout << "Start training on " << trainCount << " samples..." << std::endl;
    network.train(trainInputs, trainTargets, 0.01, 1, 32, true);

    return 0;
}
