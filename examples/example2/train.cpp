#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "QNet.hpp"
using namespace std;

using Mat = QMath::Matrix<float>;

#define os cout
// ofstream os("train.log");
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
    QNet::Net<float, Mat, QNet::sigmoid, QNet::dSigmoid, true, true> nn(784, {256, 64, 10});
    os << "Initializing ANN..." << std::endl;
    // ann.init();
    nn.open("../examples/example2/model.model");

    vector<Mat> inputs, targets;
    os << "Loading training data..." << std::endl;
    loadData("../examples/example2/trainData.txt", inputs, targets);
    os << "Training..." << std::endl;
    size_t sum = 0;
    double learningRate;
	os << "learningRate: ";
	cin >> learningRate; 
    os << "Epochs: ";
    size_t epochs;
    cin >> epochs;
	os << "training..." << endl;
    nn.train(inputs, targets, epochs, learningRate, 1, os);
    os << "Training completed. Do you want to save the model? (y/n): ";
    char ch;
    cin >> ch;
    if (ch == 'y' || ch == 'Y') {
        os << "Saving model..." << endl;
        nn.save("../examples/example2/model.model");
        os << "Model saved." << endl;
    }
    else os << "Model not saved." << endl;
    return 0;
}
