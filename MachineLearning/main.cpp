#include "MNISTLoader.hpp"
#include "NeuralNetworkSoftmax.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

namespace {

struct TrainingSet {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
};

struct TestSet {
    std::vector<std::vector<double>> inputs;
    std::vector<int> labels;
};

struct EvaluationResult {
    size_t correct = 0;
    size_t total = 0;
    double accuracy = 0.0;
};

TrainingSet prepareTrainingSet(const std::vector<std::vector<double>> &images,
                               const std::vector<int> &labels,
                               size_t limit) {
    TrainingSet dataset;
    const size_t sampleCount = std::min<size_t>(limit, images.size());

    dataset.inputs.reserve(sampleCount);
    dataset.targets.reserve(sampleCount);

    for (size_t i = 0; i < sampleCount; ++i) {
        dataset.inputs.push_back(images[i]);
        dataset.targets.push_back(labelToOneHot(labels[i]));
    }

    return dataset;
}

TestSet prepareTestSet(const std::vector<std::vector<double>> &images,
                       const std::vector<int> &labels) {
    TestSet dataset;
    dataset.inputs = images;
    dataset.labels = labels;
    return dataset;
}

std::vector<Layer> buildLayers() {
    std::vector<Layer> layers;
    layers.emplace_back(0, 784, ActivationFunctionType::None);
    layers.emplace_back(1, 128, ActivationFunctionType::RELU);
    layers.emplace_back(2, 64, ActivationFunctionType::RELU);
    layers.emplace_back(3, 10, ActivationFunctionType::None);
    return layers;
}

NeuralNetworkSoftmax trainNetwork(const TrainingSet &trainingSet) {
    NeuralNetworkSoftmax network(buildLayers());
    std::cout << "Start training on " << trainingSet.inputs.size()
              << " samples..." << std::endl;
    network.train(trainingSet.inputs, trainingSet.targets, 0.01, 10, 32, true);
    return network;
}

EvaluationResult evaluateAccuracy(NeuralNetworkSoftmax &network,
                                  const TestSet &testSet) {
    EvaluationResult result;
    result.total = testSet.inputs.size();

    for (size_t i = 0; i < result.total; ++i) {
        const size_t predictedLabel = network.predictClass(testSet.inputs[i]);

        if (predictedLabel == static_cast<size_t>(testSet.labels[i])) {
            ++result.correct;
        }
    }

    result.accuracy = result.total == 0
                          ? 0.0
                          : static_cast<double>(result.correct) / result.total;
    return result;
}

} // namespace

int main() {
    std::cout << "Loading Fashion-MNIST..." << std::endl;

    auto trainImages = readImages("data/train-images-idx3-ubyte");
    auto trainLabels = readLabels("data/train-labels-idx1-ubyte");
    auto testImages  = readImages("data/t10k-images-idx3-ubyte");
    auto testLabels  = readLabels("data/t10k-labels-idx1-ubyte");

    TrainingSet trainingSet = prepareTrainingSet(trainImages, trainLabels, trainImages.size());
    TestSet testSet = prepareTestSet(testImages, testLabels);

    std::cout << "Train: " << trainImages.size() << " images" << std::endl;
    std::cout << "Test:  " << testImages.size()  << " images" << std::endl;
    std::cout << "First image size: " << trainImages[0].size() << " pixels" << std::endl;
    std::cout << "First label: " << trainLabels[0] << std::endl;

    NeuralNetworkSoftmax network = trainNetwork(trainingSet);

    const EvaluationResult evaluation = evaluateAccuracy(network, testSet);
    std::cout << "Test accuracy: " << evaluation.accuracy * 100.0 << "% ("
              << evaluation.correct << "/" << evaluation.total << ")"
              << std::endl;

    return 0;
}
