#ifndef SOFTMAX_CROSS_ENTROPY_HPP
#define SOFTMAX_CROSS_ENTROPY_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SoftmaxCrossEntropy {

inline std::vector<double> softmax(const std::vector<double> &logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Softmax input must not be empty.");
    }

    const double maxValue = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size(), 0.0);

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - maxValue);
        sum += probabilities[i];
    }

    if (sum <= 0.0) {
        throw std::runtime_error("Softmax normalization failed.");
    }

    for (double &value : probabilities) {
        value /= sum;
    }

    return probabilities;
}

inline double crossEntropyLoss(const std::vector<double> &probabilities,
                               const std::vector<double> &targets,
                               double epsilon = 1e-12) {
    if (probabilities.empty() || targets.empty()) {
        throw std::invalid_argument("Cross entropy inputs must not be empty.");
    }
    if (probabilities.size() != targets.size()) {
        throw std::invalid_argument(
            "Probabilities and targets must have the same size.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        const double clippedProbability = std::min(std::max(probabilities[i], epsilon), 1.0 - epsilon);
        loss -= targets[i] * std::log(clippedProbability);
    }
    return loss;
}

inline double softmaxCrossEntropyLoss(const std::vector<double> &logits,
                                      const std::vector<double> &targets,
                                      double epsilon = 1e-12) {
    return crossEntropyLoss(softmax(logits), targets, epsilon);
}

inline std::vector<double> softmaxCrossEntropyGradient(
    const std::vector<double> &logits,
    const std::vector<double> &targets) {
    if (logits.empty() || targets.empty()) {
        throw std::invalid_argument("Gradient inputs must not be empty.");
    }
    if (logits.size() != targets.size()) {
        throw std::invalid_argument(
            "Logits and targets must have the same size.");
    }

    std::vector<double> probabilities = softmax(logits);
    for (size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] -= targets[i];
    }
    return probabilities;
}

inline size_t predictClass(const std::vector<double> &probabilities) {
    if (probabilities.empty()) {
        throw std::invalid_argument("Prediction input must not be empty.");
    }

    size_t bestIndex = 0;
    for (size_t i = 1; i < probabilities.size(); ++i) {
        if (probabilities[i] > probabilities[bestIndex]) {
            bestIndex = i;
        }
    }
    return bestIndex;
}

inline size_t predictClassFromLogits(const std::vector<double> &logits) {
    return predictClass(softmax(logits));
}

} // namespace SoftmaxCrossEntropy

#endif // SOFTMAX_CROSS_ENTROPY_HPP
