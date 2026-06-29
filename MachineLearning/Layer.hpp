#ifndef Layer_hpp
#define Layer_hpp

#include <cmath>      //定义数学函数
#include <functional> //定义函数对象
#include <stdexcept>
#include <vector> //定义可变长数组

using Activating_Function = std::function<double(double)>; // 自定义激活函数类型

namespace ActivationFunctions {
inline double relu(double x) { return x > 0 ? x : 0; } // 定义ReLU函数
inline double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
} // 定义ReLU函数的导数
inline double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
} // 定义Sigmoid函数
inline double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
} // 定义Sigmoid函数的导数
} // namespace ActivationFunctions

enum class ActivationFunctionType {
    None,
    RELU,
    SIGMOID
}; // 定义激活函数类型枚举

inline std::pair<Activating_Function, Activating_Function>

getActivationFunction(ActivationFunctionType type) {
    using namespace ActivationFunctions;
    switch (type) {
    case ActivationFunctionType::RELU:
	return {ActivationFunctions::relu, ActivationFunctions::reluDerivative};
    case ActivationFunctionType::SIGMOID:
	return {ActivationFunctions::sigmoid,
		ActivationFunctions::sigmoidDerivative};
    case ActivationFunctionType::None:
    default:
	return {Activating_Function{}, Activating_Function{}};
    }
} // 根据激活函数类型返回对应的激活函数和导数

struct Layer // NN的层结构
{
  private:					      // 类自身成员函数可以访问
    Activating_Function activationFunction;	      // 激活函数
    Activating_Function activationFunctionDerivative; // 激活函数的导数

  public:
    int lay_index;
    int lay_size;
    std::vector<double> Z_input;  // 输入向量
    std::vector<double> A_output; // 输出向量
    std::vector<double> W;	  // 权重向量
    std::vector<double> bias;	  // 偏置向量
    std::vector<double> gradient; // 梯度向量

    Layer(int index, int size,ActivationFunctionType activationType) 
    : lay_index(index), lay_size(size), Z_input(size, 0.0),A_output(size, 0.0) 
      { // 初始化变量列表
	if (size <= 0) {
	    throw std::invalid_argument("Layer size must be positive.");
	}
	if (index != 0) {
	    gradient = std::vector<double>(size, 0.0);//初始化变量
	    bias = std::vector<double>(size, 0.0);//初始化变量
	    activationFunction = getActivationFunction(activationType).first;//pair.first是激活函数，pair.second是激活函数的导数
	    activationFunctionDerivative = getActivationFunction(activationType).second;
	};
    }//构造函数：实现

    double applyActivationFunction(double x) const {
	if (!activationFunction) {
	    throw std::runtime_error(
		"This layer does not have an activation function.");
	};
    return activationFunction(x);
    };

    double applyActivationFunctionDerivative(double x) const {
	if (!activationFunctionDerivative) {
	    throw std::runtime_error(
		"This layer does not have an activation function derivative.");
	};
	return activationFunctionDerivative(x);
    };

    bool hasActivationFunction() const {
	return static_cast<bool>(activationFunction);
    };
    bool hasActivationFunctionDerivative() const {
	return static_cast<bool>(activationFunctionDerivative);
    };
};
#endif // Layer_hpp,关闭头文件保护