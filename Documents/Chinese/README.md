# QNet

## 简介

本项目是一个 Head-Only 库，其主要内容位于 QNet。其中实现了一个神经网络类模板，支持定制简单的由线性层组成的神经网络，并提供自动微分功能。可以通过类模板定制网络的层数，输入层输出层神经元数，以及各个隐藏层的神经元数量，还可以定制激活函数。

## 依赖

本项目依赖于 QMath 项目中的 `LinearAlgebra.cpp` 文件。务必保证其处于相应文件夹或手动修改文件 `QNet.hpp` 中的相关 `#include`。

## 使用说明

本项目提供了两个 Example，位于 `examples` 文件夹中， 两个案例均借助了 CMake 构建。 

### Example1

本 example 通过神经网络实现了数字比大小。即：输入两个数并比较它们的大小。

可以在 `examples/example1/example1.cpp` 中查看完整代码。

在 example1 中，以下代码用于构建一个神经网络并初始化权重：

```cpp
typedef QMath::Matrix<float> Mat;

QNet::Net<float, Mat, QNet::sigmoid, QNet::dSigmoid, true, true> nn(2, {4, 2, 1});
nn.init();
```

解释：
  * 该神经网络中的参数均基于 `float` 存储。
  * Mat 为矩阵乘法模板类，编来源于 QMath 的 `LinearAlgebra.hpp`。注：我突然发现第二个模板参数是不必要的，因为第一个模板参数足以推导出第二个，但出于学业压力，没时间修改，故暂时保留，见谅。
  * 第三第四个模板参数分别为激活函数和激活函数的导数，激活函数会在每个线性层的层末应用。
  * 第五个参数表示：是否为所有线性层开启偏置（bias）。要么全部开启要么不开启，不支持单独开启。
  * 最后一个模板参数代表：是否对输出层应用激活函数。
  * 在构造函数中传入了 `(2, {4, 2, 1})`，代表共有三个线性层，输入层神经元数为 $2$，两个隐藏层的神经元个数分别为 $4,2$，输出层的神经元数为 $1$。
  * `nn.init()` 用于初始化神经网络权重。

这段代码用于生成训练数据，生成方式很简单，随机两个数字并比较它们的大小关系：

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

这段代码用于训练：

```cpp
nn.train(inputs, targets, 1000, 1e-2, 100, os);
nn.train(inputs, targets, 500, 1e-3, 100, os);
```

调用 `nn.train` 即可训练，需要提供输入数据，目标答案，训练轮数和学习率。同时，倒数第二个参数指定了每多少轮训练打印一次训练信息，最后一个参数接受一个输出流 `std::ofstream` 用于打印训练信息。无需提供损失函数。

最后一段代码现场生成了测试数据，并调用 `nn.forward` 前向传播，最后计算 accuracy：

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

以上为针对 example1 的全部解释。

## Example2

exmaple2 使用 MNIST 数据集实现了手写数字识别。其用到的 QNet 功能与 example1 基本相同。无非是网络层数更多，神经元数量更多。与 example1 不同的是，它使用 `nn.save`、`nn.open` 实现了模型的读取与保存。

可以在 `examples/example2/` 中查看完整代码。

example2 提供了一个训练好的模型 `examples/example2/model.model`，其 accuracy 约为 95%。