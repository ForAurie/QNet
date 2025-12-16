#include "QNet.hpp"
#include <iostream>
#include <vector>
typedef LinearAlgebra::Matrix<float> Mat;
using namespace std;
constexpr size_t sum = 1000;
/*
template<typename T = float,
         typename Mat = LinearAlgebra::Matrix<T>,
         T (*actFunc)(const T&) = sigmoid<T>,
         T (*dActFunc)(const T&) = dSigmoid<T>,
         const bool haveBiases = true,
         const bool haveOutputActivation = true>
ANN(const size_t& __inputSize, const size_t& __outputSizNet.hppe, const std::vector<size_t>& hiddenLayers)
 */
#define os cout
int main() {
    QANN::ANN<float, Mat, QANN::sigmoid, QANN::dSigmoid, true, true> ann(2, 1, {4, 2});
    ann.init();
    
    vector<Mat> inputs, targets;
    for (size_t i = 0; i < sum; i++) {
        float x = QANN::randomReal<float>(-1.0, 1.0);
        float y = QANN::randomReal<float>(-1.0, 1.0);
        Mat tmp(1, 2, 0);
        tmp(0, 0) = x; tmp(0, 1) = y;
        inputs.push_back(Mat({{x, y}}));
        tmp.resize(1, 1, x < y);
        targets.push_back(tmp);
    }
    ann.train(inputs, targets, 1000, 1e-2, 100, os); // 训练一千轮，学习率 1e-2，每 100 轮向流 os 输出一次进度
    ann.train(inputs, targets, 500, 1e-3, 100, os);
    inputs.clear(); targets.clear();
    size_t correct = 0;
    for (size_t i = 0; i < sum; i++) {
        float x = QANN::randomReal<float>(-1.0, 1.0);
        float y = QANN::randomReal<float>(-1.0, 1.0);
        Mat tmp(1, 2, 0);
        tmp(0, 0) = x; tmp(0, 1) = y;
        auto res = ann.forward(tmp);
        if ((int) round(res(0, 0)) == (int) (x < y)) correct++;
    }
    // ann.save("example1.model");
    os << "Accuracy: " << correct << " / " << sum << " = " << (double) correct / sum * 100.0 << "%" << endl;
    return 0;
}