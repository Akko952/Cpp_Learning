#include "MNISTLoader.hpp"
#include "NeuralNetworkSoftmax.hpp"
#include <algorithm>
#include <iostream>
#include <string>
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

std::vector<Layer> buildLayers() {//调整网络结构
    std::vector<Layer> layers;
    layers.emplace_back(0, 784, ActivationFunctionType::None);
    layers.emplace_back(1, 256, ActivationFunctionType::RELU);
    layers.emplace_back(2, 10, ActivationFunctionType::None);
    return layers;
}

NeuralNetworkSoftmax trainNetwork(const TrainingSet &trainingSet, size_t start_epoch = 0) {
    NeuralNetworkSoftmax network(buildLayers());

    const size_t additional_epochs = (start_epoch > 0) ? 1 : 10;
    const size_t total_epochs = start_epoch + additional_epochs;

    if (start_epoch > 0) {
        std::string checkpoint = "model_epoch_" + std::to_string(start_epoch) + ".csv";
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  RESUME MODE: Loaded " << checkpoint << std::endl;
        std::cout << "  Training " << additional_epochs << " additional epochs ("
                  << (start_epoch + 1) << "-" << total_epochs << ")" << std::endl;
        std::cout << "============================================================\n" << std::endl;
        network.loadModel(checkpoint);
    }

    std::cout << "Start training on " << trainingSet.inputs.size()
              << " samples..." << std::endl;
    network.train(trainingSet.inputs, trainingSet.targets, 0.1,
                  total_epochs, 256, true, 0, start_epoch);
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

int main(int argc, char *argv[]) {
    // Parse --resume <epoch> argument
    size_t start_epoch = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume" && i + 1 < argc) {
            start_epoch = std::stoul(argv[++i]);
        }
    }

    if (start_epoch > 0) {
        std::cout << "=== RESUME MODE: starting from epoch " << (start_epoch + 1) << " ===" << std::endl;
    }

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

    NeuralNetworkSoftmax network = trainNetwork(trainingSet, start_epoch);

    const EvaluationResult evaluation = evaluateAccuracy(network, testSet);
    std::cout << "Test accuracy: " << evaluation.accuracy * 100.0 << "% ("
              << evaluation.correct << "/" << evaluation.total << ")"
              << std::endl;

    return 0;
}
