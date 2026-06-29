#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <random>
#include <stdexcept>
#include <vector>

class Matrix {
  private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

  public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
	data.resize(r, std::vector<double>(c, 0.0));
    };

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    double &operator()(size_t i, size_t j) { // 重载符号（）
	if (i >= getRows() || j >= getCols()) {
	    throw std::out_of_range("Matrix index out of range");
	}

	return data[i][j];
    }
    const double &operator()(size_t i, size_t j) const { // 重载常量的符号（）
	if (i >= getRows() || j >= getCols()) {
	    throw std::out_of_range("Matrix index out of range");
	}
	return data[i][j];
    };

    Matrix &fillRandom(double min = -1.0, double max = 1.0) {
	std::random_device rd;				 // 使用随机设备生成种子
	std::mt19937 gen(rd());				 // 使用随机数生成器
	std::uniform_real_distribution<> dist(min, max); // 定义均匀分布

	for (size_t i = 0; i < (*this).getRows(); ++i) {
	    for (size_t j = 0; j < getCols(); ++j) {
		(*this)(i, j) = dist(gen); // 填充随机数
	    };
	}
	return *this;
    };

    Matrix operator*(double scalar) const { // 矩阵点乘，新一个matrix
	Matrix result(getRows(), getCols());
	for (size_t i = 0; i < getRows(); ++i) {
	    for (size_t j = 0; j < getCols(); ++j) {
		result(i, j) = (*this)(i, j) * scalar;
	    }
	}
	return result;
    }

    Matrix &operator*=(double scalar) { // 直接变自身
	for (size_t i = 0; i < getRows(); ++i) {
	    for (size_t j = 0; j < getCols(); ++j) {
		(*this)(i, j) *= scalar;
	    }
	}
	return *this;
    };
};
#endif // MATRIX_HPP