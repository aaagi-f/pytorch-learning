# MNIST手写数字识别

## 这是什么
用PyTorch搭建的3层神经网络，识别0-9的手写数字。

## 为什么做这个项目
暑假自学PyTorch的第一个实战项目，熟悉：
- 神经网络怎么搭（Linear + ReLU）
- 数据怎么加载（MNIST数据集）
- 模型怎么训练（反向传播 + 优化器）
- 模型怎么保存和加载
- 不同优化器的效果对比（SGD vs Adam）

## 运行环境
- Python 3.x
- PyTorch
- torchvision

安装命令：
    pip install torch torchvision

## 文件说明
- mnist_train.py：最基础的训练脚本，SGD优化器，准确率92.64%
- train_v2.py：进阶训练脚本，Adam优化器，含验证+自动保存最佳模型，准确率96.45%
- predict.py：推理脚本，加载模型预测单张图片
- best_mnist_model.pth：训练好的模型权重文件（Adam，96.45%版本）

## 运行方式
基准版（SGD）：
    python mnist_train.py

进阶版（Adam，推荐）：
    python train_v2.py

预测：
    python predict.py

## 训练结果对比

配置        | 优化器 | 学习率 | 5轮最佳准确率
-----------|--------|--------|-------------
基准版      | SGD    | 0.001  | 92.64%
进阶版      | Adam   | 0.001  | 96.45%

Adam训练日志：
- Epoch 1：93.33%
- Epoch 2：95.35%
- Epoch 3：96.07%
- Epoch 4：96.45%（最佳）
- Epoch 5：95.08%

## 推理验证
    $ python predict.py
    测试集第0张图，真实标签: 7，模型预测: 7

## 观察
- Adam收敛速度明显快于SGD，第一轮验证即达93.33%
- SGD第一轮训练loss还在2.1，Adam已降至0.76
- 最终Adam比SGD高3.81个百分点
