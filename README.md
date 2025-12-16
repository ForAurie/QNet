# QNet

## 前言

本项目依赖于 QMath 项目中的 `LinearAlgebra` 文件。务必保证其处于相应文件夹或手动需改文件 `QANN` 中的相关 `#include`。

## 训练数据

训练数据在 `Data.7z` 中，解压后即可使用。


## 示例

两个示例分别在 `example1.cpp`、`example2-train.cpp & example2-test.cpp`。运行它们之前请确保 `#include` 路径合法且将 `Data.7z` 中的 `TrainData.txt`、`TestData.txt` 置于同一文件夹内。 特别的，`example2.model` 是已经训练好的第二个示例的模型，正确率约 $95\%$。
