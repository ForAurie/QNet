# QNet

## Introduction

This project is a head-only library, with its main content located in `QNet`. It implements a neural network class template that supports customizing simple neural networks composed of linear layers and provides automatic differentiation functionality. You can customize the number of layers, the number of neurons in the input and output layers, the number of neurons in each hidden layer, and the activation function through the class template.

## Dependencies

This project depends on the `LinearAlgebra.cpp` file from the QMath project. Make sure it is placed in the corresponding folder, or manually modify the relevant `#include` directives in the file `QNet.hpp`.

## Usage Instructions

This project provides two examples, located in the `examples` folder. Both examples are built using CMake.

### Example 1

This example implements number comparison using a neural network. That is, it takes two numbers as input and compares their magnitudes.

You can view the complete code in `examples/example1/example1.cpp`.

In example1, the following code is used to construct a neural network and initialize its weights:

```cpp
typedef QMath::Matrix<float> Mat;

QNet::Net<float, Mat, QNet::sigmoid, QNet::dSigmoid, true, true> nn(2, {4, 2, 1});
nn.init();
```

Explanation:
  * All parameters in this neural network are stored based on `float`.
  * `Mat` is a matrix multiplication template class, originating from QMath's `LinearAlgebra.hpp`. Note: I suddenly realized that the second template parameter is unnecessary, because the first template parameter is sufficient to deduce the second. However, due to academic pressure, I haven't had time to modify it, so it is temporarily retained. Please excuse this.
  * The third and fourth template parameters are the activation function and its derivative, respectively. The activation function is applied at the end of each linear layer.
  * The fifth parameter indicates whether bias is enabled for all linear layers. It is either enabled for all layers or disabled for all; individual layer control is not supported.
  * The last template parameter indicates whether the activation function is applied to the output layer.
  * The constructor is passed `(2, {4, 2, 1})`, which means there are three linear layers in total: the input layer has $2$ neurons, the two hidden layers have $4$ and $2$ neurons respectively, and the output layer has $1$ neuron.
  * `nn.init()` is used to initialize the neural network weights.

The following code is used to generate training data. The generation method is straightforward: randomly generate two numbers and compare their magnitudes:

```cpp
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
```

The following code is used for training:

```cpp
nn.train(inputs, targets, 1000, 1e-2, 100, os);
nn.train(inputs, targets, 500, 1e-3, 100, os);
```

Calling `nn.train` performs the training. You need to provide input data, target answers, the number of training epochs, and the learning rate. Additionally, the second-to-last parameter specifies how often (every how many epochs) training information is printed, and the last parameter accepts an output stream `std::ofstream` for printing training information. No loss function needs to be provided.

The final piece of code generates test data on the fly, calls `nn.forward` for forward propagation, and finally computes the accuracy:

```cpp
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
```

The above covers the complete explanation for example1.

## Example 2

Example 2 implements handwritten digit recognition using the MNIST dataset. The QNet functionality it uses is basically the same as in example1, except that the network has more layers and more neurons. The difference from example1 is that it uses `nn.save` and `nn.open` to implement model loading and saving.

You can view the complete code in `examples/example2/`.

Example 2 provides a pre-trained model `examples/example2/model.model`, which achieves an accuracy of approximately 95%.