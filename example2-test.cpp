#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "QNet.hpp"
using namespace std;
using Mat = LinearAlgebra::Matrix<float>;
#define os cout

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

int predict(const Mat& output) {
    int maxIndex = 0;
    for (size_t i = 1; i < output.M(); i++)
        if (output(0, i) > output(0, maxIndex))
            maxIndex = i;
    return maxIndex;
}

int main() {
    os << "Loading ANN model..." << endl;
    QANN::ANN<float, Mat, QANN::sigmoid, QANN::dSigmoid, true, true> ann(784, 10, {256, 64});
    cout << "Path: ";
	string Path;
	cin >> Path;
	cout << endl;
	ann.open(Path);

    os << "Loading Testing data..." << endl;
    vector<Mat> inputs, targets;
    loadData(".\\TestData.txt", inputs, targets);

    os << "Testing..." << endl;
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        correct += (predict(ann.forward(inputs[i])) == predict(targets[i]));
    }
    os << "Testing completed." << endl;
    os << "Accuracy: " << correct << " / " << inputs.size() << " = " << (correct * 100.0 / inputs.size()) << "%" << endl;
    int tmp; while (cin >> tmp);
    return 0;
}
