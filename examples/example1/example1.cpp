#include "QNet.hpp"
#include <iostream>
#include <vector>
typedef QMath::Matrix<float> Mat;
using namespace std;
constexpr size_t sum = 1000;
#define os   cout
int main() {
    QNet::Net<float, Mat, QNet::sigmoid, QNet::dSigmoid, true, true> nn(2, {4, 2, 1});
    nn.init();
    
    vector<Mat> inputs, targets;
    for (size_t i = 0; i < sum; i++) {
        float x = QNet::randomReal<float>(-1.0, 1.0);
        float y = QNet::randomReal<float>(-1.0, 1.0);
        Mat tmp(1, 2, 0);
        tmp(0, 0) = x; tmp(0, 1) = y;
        inputs.push_back(Mat({{x, y}}));
        tmp.assgin(1, 1, x < y);
        targets.push_back(tmp);
    }
    nn.train(inputs, targets, 1000, 1e-2, 100, os); // 训练一千轮，学习率 1e-2，每 100 轮向流 os 输出一次进度
    nn.train(inputs, targets, 500, 1e-3, 100, os);
    inputs.clear(); targets.clear();
    size_t correct = 0;
    for (size_t i = 0; i < sum; i++) {
        float x = QNet::randomReal<float>(-1.0, 1.0);
        float y = QNet::randomReal<float>(-1.0, 1.0);
        Mat tmp(1, 2, 0);
        tmp(0, 0) = x; tmp(0, 1) = y;
        auto res = nn.forward(tmp);
        if ((int) round(res(0, 0)) == (int) (x < y)) correct++;
    }
    os << "Accuracy: " << correct << " / " << sum << " = " << (double) correct / sum * 100.0 << "%" << endl;
    return 0;
}