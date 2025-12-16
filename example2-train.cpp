#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "QNet.hpp"
using namespace std;

using Mat = LinearAlgebra::Matrix<float>;

// #define os cout
ofstream os("train.log");
void loadData(const string& Path, vector<Mat>& inputs, vector<Mat>& targets) {
    ifstream fin(Path);
    int tmp;
    for (size_t i = 0; i < 10; i++) {
        size_t sz; Mat ans(1, 10, 0), in(1, 28 * 28, 0); ans(0, i) = 1;
        fin >> i >> sz;
        os << "Loading " << sz << " samples for digit " << i << endl;
        while (sz--) {
            for (size_t j = 0; j < 28 * 28; j++)
                fin >> tmp, in(0, j) = tmp;
            inputs.push_back(in);
            targets.push_back(ans);
        }
    }
    fin.close();
}

int main() {
    QANN::ANN<float, Mat, QANN::sigmoid, QANN::dSigmoid, true, true> ann(784, 10, {256, 64});
    os << "Initializing ANN..." << std::endl;
    // ann.init();
    ann.open("example2.model");

    vector<Mat> inputs, targets;
    os << "Loading training data..." << std::endl;
    loadData(".\\TrainData.txt", inputs, targets);
    os << "Training..." << std::endl;
    size_t sum = 0;
    double learningRate;
	cout << "learningRate: ";
	cin >> learningRate; 
	cout << "training..." << endl;
    while (1) {
        ann.train(inputs, targets, 10 , learningRate, 1, os);
        ann.save("example2_res.model");
        sum += 10;
        os << "\nsum = " << sum << endl << endl;
    }
    // os << "Training completed." << endl;
    return 0;
}
