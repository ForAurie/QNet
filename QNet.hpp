#include "LinearAlgebra.hpp"
#include <random>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <ctime>
namespace QNet {
    template<typename T>
    T randomReal(const T& l, const T& r) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(l, r);
        return dis(gen);
    }
    template<typename T>
    inline T sigmoid(const T& x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    template<typename T>
    inline T dSigmoid(const T& x) {
        const T E = std::exp(-x);
        const T t = 1 + E;
        return E / (t * t);
    }
    template<typename T = float, typename Mat = LinearAlgebra::Matrix<T>, T (*actFunc)(const T&) = sigmoid<T>, T (*dActFunc)(const T&) = dSigmoid<T>, const bool haveBiases = true, const bool haveOutputActivation = true>
    class Net {
    private:
        std::vector<Mat> weights, biases;
        std::vector<std::pair<Mat, Mat>> __forward(Mat input) const {
            std::vector<std::pair<Mat, Mat>> res;
            for (size_t i = 0; i < weights.size() - 1; i++) {
                input *= weights[i];
                if (haveBiases) input += biases[i];
                res.push_back(std::make_pair(input, Mat()));
                input.applyFunctionSelf(actFunc);
                res.back().second = input;
            }
            input *= weights.back();
            if (haveBiases) input += biases.back();
            res.push_back(std::make_pair(input, Mat()));
            if (haveOutputActivation) {
                input.applyFunctionSelf(actFunc);
                res.back().second = input;
            }
            return res;
        }
    public:
        size_t getInputSize() const { return weights.front().N(); }
        size_t getOutputSize() const { return weights.back().M(); }
        std::vector<Mat>& getWeights() { return weights; }
        std::vector<Mat>& getBiases() { return biases; }
        const std::vector<Mat>& getWeights() const { return weights; }
        const std::vector<Mat>& getBiases() const { return biases; }
        Net(size_t inputSize, const std::vector<size_t>& Layers): weights(Layers.size()), biases(0) {
            if (haveBiases) biases.resize(Layers.size());
            weights.front().resize(inputSize, Layers.front());
            if (haveBiases) biases.front().resize(1, Layers.front());
            for (size_t i = 1; i < Layers.size(); ++i) {
                weights[i].resize(Layers[i - 1], Layers[i]);
                if (haveBiases) biases[i].resize(1, Layers[i]);
            }
        }
        Mat forward(Mat input) const {
            for (size_t i = 0; i + 1 < weights.size(); ++i) {
                input *= weights[i];
                if (haveBiases) input += biases[i];
                input.applyFunctionSelf(actFunc);
            }
            input *= weights.back();
            if (haveBiases) input += biases.back();
            if (haveOutputActivation) input.applyFunctionSelf(actFunc);
            return input;
        }
        void init(const T& l = -1.0, const T& r = 1.0, T (*randomFunc)(const T&, const T&) = randomReal<T>) {
            for (auto &x: weights)
                for (size_t i = 0; i < x.N(); ++i)
                    for (size_t j = 0; j < x.M(); ++j)
                        x(i, j) = randomFunc(l, r);
            if (!haveBiases) return;
            for (auto &x: biases)
                for (size_t i = 0; i < x.N(); ++i)
                    for (size_t j = 0; j < x.M(); ++j)
                        x(i, j) = randomFunc(l, r);
        }
        void save(const std::string& filename) const {
            FILE *fout = fopen(filename.c_str(), "wb");
            char sign = 'Q';
            unsigned int typeSize = sizeof(T);
            fwrite(&sign, sizeof(char), 1, fout); // 标志
            fwrite(&typeSize, sizeof(unsigned int), 1, fout); // 数据类型大小
            bool __haveBiases = haveBiases, __haveOutputActivation = haveOutputActivation;
            fwrite(&__haveBiases, sizeof(bool), 1, fout); // 是否有偏置
            fwrite(&__haveOutputActivation, sizeof(bool), 1, fout); // 是否有输出激活函数
            size_t layers = weights.size(); // 隐藏层加输出层数量
            fwrite(&layers, sizeof(size_t), 1, fout);
            T tmp;
            for (const auto &x: weights) {
                size_t n = x.N(), m = x.M();
                fwrite(&n, sizeof(size_t), 1, fout);
                fwrite(&m, sizeof(size_t), 1, fout);
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < m; j++) {
                        tmp = x(i, j);
                        fwrite(&tmp, sizeof(T), 1, fout);
                    }
                }
            }
            if (haveBiases) {
                for (const auto &x: biases) {
                    size_t n = x.M();
                    fwrite(&n, sizeof(size_t), 1, fout);
                    for (size_t j = 0; j < n; j++) {
                        tmp = x(0, j);
                        fwrite(&tmp, sizeof(T), 1, fout);
                    }
                }
            }
            fclose(fout);
        }
        void open(const std::string& filename) {
            FILE *fin = fopen(filename.c_str(), "rb");
            char sign;
            unsigned int typeSize;
            fread(&sign, sizeof(char), 1, fin);
            assert(sign == 'Q'); // 检查标志
            fread(&typeSize, sizeof(unsigned int), 1, fin);
            assert(typeSize == sizeof(T)); // 检查数据类型大小
            bool __haveBiases, __haveOutputActivation;
            fread(&__haveBiases, sizeof(bool), 1, fin);
            fread(&__haveOutputActivation, sizeof(bool), 1, fin);
            assert(__haveBiases == haveBiases && __haveOutputActivation == haveOutputActivation); // 检查偏置和输出层激活函数一致性
            size_t layers;
            fread(&layers, sizeof(size_t), 1, fin);
            weights.resize(layers);
            if (haveBiases) biases.resize(layers);
            T tmp;
            for (auto &x: weights) {
                size_t n, m;
                fread(&n, sizeof(size_t), 1, fin);
                fread(&m, sizeof(size_t), 1, fin);
                x.resize(n, m);
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < m; j++) {
                        fread(&tmp, sizeof(T), 1, fin);
                        x(i, j) = tmp;
                    }
                }
            }
            if (haveBiases) {
                for (auto &x: biases) {
                    size_t n;
                    fread(&n, sizeof(size_t), 1, fin);
                    x.resize(1, n);
                    for (size_t i = 0; i < n; i++) {
                        fread(&tmp, sizeof(T), 1, fin);
                        x(0, i) = tmp;
                    }
                }
            }
            fclose(fin);
        }
        void train(const std::vector<Mat>& inputs, const std::vector<Mat>& targets, const size_t& epochs, const T& lRate, const size_t& len = 0, std::ostream& os = std::cout) {
            std::vector<Mat> errors(weights.size());
            clock_t start = clock();
            clock_t st = clock();
            for (size_t i = 0; i < errors.size(); i++) errors[i].resize(1, weights[i].M());
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                if (len > 0 && epoch % len == 0) {
                    os << "Epoch: " << epoch << " started..." << std::endl;
                    start = clock();
                }
                for (size_t id = 0; id < inputs.size(); id++) {
                    auto output = __forward(inputs[id]);
                    if (haveOutputActivation) {
                        errors.back() = (output.back().second - targets[id]) % output.back().first.applyFunction(dActFunc);
                        // for (size_t i = 0; i < outputSize; i++)
                            // errors.back()(0, i) = (output.back().second(0, i) - targets[id](0, i)) * dActFunc(output.back().first(0, i));
                    } else errors.back() = output.back().first - targets[id];
                    for (size_t i = weights.size() - 1; i > 0; i--) {
                        errors[i - 1] = errors[i] * weights[i].transpose();
                        errors[i - 1] %= output[i - 1].first.applyFunction(dActFunc);
                        // for (size_t j = 0; j < errors[i - 1].M(); j++)
                            // errors[i - 1](0, j) *= dActFunc(output[i - 1].first(0, j));
                        if (haveBiases) biases[i] -= errors[i] * lRate;
                        for (size_t k = 0; k < weights[i].N(); k++)
                            for (size_t j = 0; j < weights[i].M(); j++)
                                weights[i](k, j) -= lRate * errors[i](0, j) * output[i - 1].second(0, k);
                    }
                    if (haveBiases) biases.front() -= errors.front() * lRate;
                    for (size_t j = 0; j < weights.front().N(); j++)
                        for (size_t i = 0; i < weights.front().M(); i++)
                            weights.front()(j, i) -= lRate * errors.front()(0, i) * inputs[id](0, j);
                }
                if (len > 0 && (epoch + 1) % len == 0) {
                    os << "Epoch: " << epoch << " completed." << std::endl;
                    os << "Time elapsed: " << double(clock() - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
                    const double dt = double(clock() - start) / CLOCKS_PER_SEC / len * 1000;
                    os << dt << " ms per epoch, Sum: " << epochs * dt / 1000 << " s, Actually sum: " << double(clock() - st) / CLOCKS_PER_SEC << ", Need: " << (epochs - epoch - 1) * dt / 1000 << " s." << std::endl;
                    os << std::endl;
                }
            }
        }
    };
}