# NiN
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result


---

## 简介
NiN（Network in Network）是由 Min Lin、Qiang Chen 与 Shuicheng Yan 于 2013 年提出的革命性深度卷积神经网络，相关成果发表于《Network In Network》。它突破了传统 CNN“卷积层堆叠+全连接层分类”的单一范式，首次提出**MLPConv（微网络卷积）** 模块与**全局平均池化**两大核心设计，用更灵活的非线性特征提取器替代了线性卷积核，同时彻底消除了全连接层带来的巨大参数量与过拟合风险。NiN 不仅将 CIFAR-10 数据集的分类错误率大幅降低至 10.41%（当时最优），更开创了 1×1 卷积在深度学习中的广泛应用，为后续 GoogLeNet、ResNet、MobileNet 等经典轻量级与高性能模型奠定了关键技术基础。

## 架构
NiN 的核心架构为**全卷积式端到端深度神经网络**，整体由**堆叠的 MLPConv 特征提取模块**与**全局平均池化分类模块**构成，完全摒弃了传统的全连接层。原论文标准输入为 224×224 分辨率的 3 通道 RGB 图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
- **特征提取模块（4 个 MLPConv 块）**：每个 MLPConv 块由 1 个标准卷积层 + 2 个连续的 1×1 卷积层组成，其中 1×1 卷积层模拟了全连接层的非线性特征组合能力，能够在保持空间结构的同时实现跨通道的信息交互。前 3 个 MLPConv 块后均接 3×3 步长 2 的重叠最大池化层进行尺寸压缩，逐步提取从边缘、纹理到深层语义的多级特征。
- **分类输出模块（全局平均池化）**：最后一个 MLPConv 块的输出通道数直接设置为分类任务的类别数（原论文 ImageNet 任务为 1000 维），随后通过全局平均池化将每个通道的特征图平均为一个标量，直接对应各类别的预测得分，无需展平操作与全连接层。

该架构通过 MLPConv 增强了局部特征的非线性表达能力，通过全局平均池化大幅减少了参数量并天然抑制过拟合，其“用卷积替代全连接”的设计思想成为了现代轻量级 CNN 的核心准则。

<img width="461" height="171" alt="image" src="https://github.com/user-attachments/assets/3e71abf2-9657-405c-98cf-83894645f7b0" />


**注意**：我们使用的是数据集 CIFAR-10，它是 10 类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要是缩小卷积核和步长，调整通道数），但核心架构（MLPConv 模块 + 全局平均池化 + 无全连接层）完全保留。

## 数据集
我们使用的是数据集 CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为 32 × 32 ，每个类别有 6000 个图像，数据集中一共有 50000 张训练图片和 10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为 PNG 等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
NiN (Network in Network), a revolutionary deep convolutional neural network proposed in 2013 by Min Lin, Qiang Chen, and Shuicheng Yan, with its findings published in "Network In Network", broke through the single paradigm of traditional CNN "convolutional layer stacking + fully connected layer classification". It first proposed two core designs: **MLPConv (Multi-Layer Perceptron Convolution)** module and **Global Average Pooling**, replacing linear convolution kernels with more flexible nonlinear feature extractors, and completely eliminating the huge number of parameters and overfitting risks brought by fully connected layers. NiN not only significantly reduced the classification error rate on the CIFAR-10 dataset to 10.41% (the best at that time), but also pioneered the widespread application of 1×1 convolution in deep learning, laying a key technical foundation for subsequent classic lightweight and high-performance models such as GoogLeNet, ResNet, and MobileNet.

## Architecture
The core architecture of NiN is a **fully convolutional end-to-end deep neural network**, which is composed of **stacked MLPConv feature extraction modules** and a **global average pooling classification module**, completely abandoning the traditional fully connected layers. The original paper's standard input was a 224×224 resolution 3-channel RGB image, and the final output was the predicted probability of the corresponding classification category. The specific structure and design are as follows:

- **Feature Extraction Module (4 MLPConv Blocks)**: Each MLPConv block consists of 1 standard convolutional layer + 2 consecutive 1×1 convolutional layers. The 1×1 convolutional layers simulate the nonlinear feature combination ability of fully connected layers, enabling cross-channel information interaction while maintaining the spatial structure. Each of the first 3 MLPConv blocks is followed by a 3×3 overlapping max pooling layer with a stride of 2 for size compression, gradually extracting multi-level features from edges and textures to deep semantics.
- **Classification Output Module (Global Average Pooling)**: The number of output channels of the last MLPConv block is directly set to the number of categories in the classification task (1000 dimensions for the ImageNet task in the original paper). Then, global average pooling averages each channel's feature map into a scalar, which directly corresponds to the prediction score of each category, without flattening operations or fully connected layers.

This architecture enhances the nonlinear expression ability of local features through MLPConv, greatly reduces the number of parameters and naturally suppresses overfitting through global average pooling. Its design idea of "replacing fully connected layers with convolutions" has become the core criterion of modern lightweight CNNs.

<img width="461" height="171" alt="image" src="https://github.com/user-attachments/assets/ed1097d0-8df5-4878-bd05-2687a7dde9b6" />


**Note:** We use the CIFAR-10 dataset, which is a 10-class dataset. Unlike the original paper, the image size of CIFAR-10 (32×32) is much smaller than the 224×224 in the original paper. We will make minor adaptations to the network structure (mainly reducing the convolution kernels and stride, adjusting the number of channels), but the core architecture (MLPConv module + global average pooling + no fully connected layers) will be completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Hinton's students Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).
