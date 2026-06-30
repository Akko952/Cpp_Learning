#include "MNISTLoader.hpp"
#include "NeuralNetworkSoftmax.hpp"
#include <algorithm>
#include <cstdlib>
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

// ── helpers ──────────────────────────────────────────────────────────

ActivationFunctionType parseActivation(const std::string &s) {
    std::string upper;
    for (char c : s) upper += static_cast<char>(std::toupper(c));
    if (upper == "SIGMOID") return ActivationFunctionType::SIGMOID;
    return ActivationFunctionType::RELU;
}

InitMethod parseInitMethod(const std::string &s) {
    std::string lower;
    for (char c : s) lower += static_cast<char>(std::tolower(c));
    if (lower == "normal")  return InitMethod::Normal;
    if (lower == "xavier")  return InitMethod::Xavier;
    if (lower == "he")      return InitMethod::He;
    return InitMethod::Custom;
}

// ── data preparation ─────────────────────────────────────────────────

TrainingSet prepareTrainingSet(const std::vector<std::vector<double>> &images,
                               const std::vector<int> &labels,
                               size_t limit) {
    TrainingSet dataset;
    const size_t sampleCount = (limit == 0)
        ? images.size()
        : std::min<size_t>(limit, images.size());

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
    return {images, labels};
}

// ── network building ─────────────────────────────────────────────────

std::vector<Layer> buildLayers(size_t num_hiddens, size_t num_layers,
                               ActivationFunctionType activation) {
    std::vector<Layer> layers;
    layers.emplace_back(0, 784, ActivationFunctionType::None);   // input

    size_t h = num_hiddens;
    for (size_t l = 0; l < num_layers; ++l) {
        layers.emplace_back(static_cast<int>(l + 1),
                            static_cast<int>(h), activation);
        h = std::max<size_t>(10, h / 2);
    }

    layers.emplace_back(static_cast<int>(num_layers + 1), 10,
                        ActivationFunctionType::None);            // output
    return layers;
}

// ── training ─────────────────────────────────────────────────────────

NeuralNetworkSoftmax trainNetwork(const TrainingSet &trainingSet,
                                  size_t num_hiddens, size_t num_layers,
                                  ActivationFunctionType activation,
                                  double lr, size_t epochs,
                                  const std::string &prefix,
                                  InitMethod init_method,
                                  size_t start_epoch = 0) {
    auto layers = buildLayers(num_hiddens, num_layers, activation);
    NeuralNetworkSoftmax network(layers, init_method);

    const size_t additional_epochs = (start_epoch > 0) ? 1 : epochs;
    const size_t total_epochs = start_epoch + additional_epochs;

    if (start_epoch > 0) {
        std::string checkpoint = prefix + "_epoch_" + std::to_string(start_epoch) + ".csv";
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  RESUME MODE: Loaded " << checkpoint << std::endl;
        std::cout << "  Training 1 additional epoch (" << (start_epoch + 1) << ")" << std::endl;
        std::cout << "============================================================\n" << std::endl;
        network.loadModel(checkpoint);
    }

    std::cout << "Start training on " << trainingSet.inputs.size()
              << " samples..." << std::endl;
    network.train(trainingSet.inputs, trainingSet.targets, lr,
                  total_epochs, 256, true, 0, start_epoch, prefix);
    return network;
}

// ── evaluation ───────────────────────────────────────────────────────

EvaluationResult evaluateAccuracy(NeuralNetworkSoftmax &network,
                                  const TestSet &testSet) {
    EvaluationResult result;
    result.total = testSet.inputs.size();

    for (size_t i = 0; i < result.total; ++i) {
        if (network.predictClass(testSet.inputs[i])
                == static_cast<size_t>(testSet.labels[i])) {
            ++result.correct;
        }
    }

    result.accuracy = result.total == 0
                          ? 0.0
                          : static_cast<double>(result.correct) / result.total;
    return result;
}

} // namespace

// ── main ─────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    // defaults
    size_t num_hiddens   = 256;
    size_t num_layers    = 1;
    double lr            = 0.1;
    size_t epochs        = 10;
    std::string prefix   = "model";
    size_t start_epoch   = 0;
    size_t limit         = 0;
    ActivationFunctionType activation = ActivationFunctionType::RELU;
    InitMethod init_method            = InitMethod::Custom;

    // parse CLI
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--hiddens"    && i + 1 < argc) num_hiddens = std::stoul(argv[++i]);
        else if (arg == "--layers"      && i + 1 < argc) num_layers  = std::stoul(argv[++i]);
        else if (arg == "--lr"          && i + 1 < argc) lr          = std::stod(argv[++i]);
        else if (arg == "--epochs"      && i + 1 < argc) epochs      = std::stoul(argv[++i]);
        else if (arg == "--prefix"      && i + 1 < argc) prefix      = argv[++i];
        else if (arg == "--resume"      && i + 1 < argc) start_epoch = std::stoul(argv[++i]);
        else if (arg == "--limit"       && i + 1 < argc) limit       = std::stoul(argv[++i]);
        else if (arg == "--activation"  && i + 1 < argc) activation  = parseActivation(argv[++i]);
        else if (arg == "--init"        && i + 1 < argc) init_method = parseInitMethod(argv[++i]);
    }

    // banner
    std::cout << "============================================================" << std::endl;
    std::cout << "  MLP Experiment" << std::endl;
    std::cout << "  hiddens=" << num_hiddens << "  layers=" << num_layers
              << "  lr=" << lr << "  epochs=" << epochs
              << "  prefix=" << prefix << std::endl;
    std::cout << "  activation=" << (activation == ActivationFunctionType::RELU ? "ReLU" : "Sigmoid")
              << "  init=";
    switch (init_method) {
        case InitMethod::Custom: std::cout << "custom"; break;
        case InitMethod::Normal: std::cout << "normal(0,0.01)"; break;
        case InitMethod::Xavier: std::cout << "xavier"; break;
        case InitMethod::He:     std::cout << "he"; break;
    }
    std::cout << std::endl;
    if (start_epoch > 0) std::cout << "  RESUME from epoch " << start_epoch << std::endl;
    std::cout << "============================================================" << std::endl;

    // load data
    std::cout << "Loading Fashion-MNIST..." << std::endl;
    auto trainImages = readImages("data/train-images-idx3-ubyte");
    auto trainLabels = readLabels("data/train-labels-idx1-ubyte");
    auto testImages  = readImages("data/t10k-images-idx3-ubyte");
    auto testLabels  = readLabels("data/t10k-labels-idx1-ubyte");

    TrainingSet trainingSet = prepareTrainingSet(trainImages, trainLabels, limit);
    TestSet testSet = prepareTestSet(testImages, testLabels);

    std::cout << "Train: " << trainingSet.inputs.size() << " images"
              << "  Test: " << testSet.inputs.size() << " images" << std::endl;

    // train
    NeuralNetworkSoftmax network = trainNetwork(
        trainingSet, num_hiddens, num_layers, activation,
        lr, epochs, prefix, init_method, start_epoch);

    // evaluate
    const EvaluationResult evaluation = evaluateAccuracy(network, testSet);
    std::cout << "\n>>> Test accuracy: " << evaluation.accuracy * 100.0 << "%  ("
              << evaluation.correct << "/" << evaluation.total << ")" << std::endl;

    return 0;
}
