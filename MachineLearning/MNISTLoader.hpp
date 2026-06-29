#ifndef MNISTLOADER_HPP
#define MNISTLOADER_HPP

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// 读取 IDX 格式的二进制文件
// 所有数字都是大端序，需转成小端
inline int readInt(std::ifstream &file) {
    unsigned char buf[4];
    file.read((char *)buf, 4);
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

// 读取图像文件
// 返回: 每个元素是一个长度为 28*28=784 的 vector<double>，值归一化到 [0,1]
inline std::vector<std::vector<double>>
readImages(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    int magic = readInt(file);
    int count = readInt(file);
    int rows  = readInt(file);
    int cols  = readInt(file);

    std::vector<std::vector<double>> images(count,
        std::vector<double>(rows * cols));

    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            images[i][j] = pixel / 255.0;
        }
    }
    return images;
}

// 读取标签文件
// 返回: vector<int>，每个值是 0~9
inline std::vector<int> readLabels(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    int magic = readInt(file);
    int count = readInt(file);

    std::vector<int> labels(count);
    for (int i = 0; i < count; ++i) {
        unsigned char label;
        file.read((char *)&label, 1);
        labels[i] = label;
    }
    return labels;
}

// 把标签 (0~9) 转为独热编码
// 比如标签 3 → [0,0,0,1,0,0,0,0,0,0]
inline std::vector<double> labelToOneHot(int label, int numClasses = 10) {
    std::vector<double> onehot(numClasses, 0.0);
    onehot[label] = 1.0;
    return onehot;
}

#endif
