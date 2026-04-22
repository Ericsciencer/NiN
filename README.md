# VGG
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result

<img width="2480" height="1914" alt="vgg16_cifar_training_curve" src="https://github.com/user-attachments/assets/096d4f96-c10f-43ce-8e07-d6be367c819b" />


---

## 简介
VGG 是由牛津大学视觉几何组（Visual Geometry Group, VGG）的 Karen Simonyan 和 Andrew Zisserman 于 2014 年提出的经典深度卷积神经网络，相关成果发表于《Very Deep Convolutional Networks for Large-Scale Image Recognition》。它在当年的 ImageNet 大规模图像分类竞赛中，将 Top-5 错误率从 AlexNet 的 15.3% 进一步降至 7.3%，以压倒性优势获得亚军。其核心架构首次系统性地证明了**增加网络深度是提升视觉任务性能的关键因素**，并确立了"统一使用3×3小卷积核堆叠+逐层通道翻倍+最大池化降维"的标准CNN设计范式。VGG的设计思想极其简洁且具有极强的可扩展性，不仅成为了后续所有深度卷积神经网络的基础架构模板，更被广泛应用于目标检测、语义分割、图像生成等几乎所有计算机视觉领域，是深度学习发展史上最具影响力的模型之一。

## 架构
VGG的核心架构为"模块化卷积块堆叠"的端到端深度卷积神经网络，整体分为「卷积特征提取模块」和「全连接分类模块」两大核心部分。原论文提出了6种不同深度的网络变体，其中VGG16（13个卷积层+3个全连接层）和VGG19（16个卷积层+3个全连接层）是最常用的两个版本。原论文标准输入为224×224分辨率的3通道RGB图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
-  **特征提取模块（卷积层）**：由5个连续的卷积块组成，每个卷积块内部堆叠2-4个3×3卷积核（步长1，填充1），所有卷积层后均接ReLU非线性激活函数；每个卷积块末尾接一个2×2步长2的最大池化层，将特征图尺寸减半。通道数从第一个卷积块的64开始，每经过一个池化层通道数翻倍，最终达到512。这种设计使得两个3×3卷积的感受野等价于一个5×5卷积，三个3×3卷积的感受野等价于一个7×7卷积，同时大幅减少了参数量并增加了网络的非线性表达能力。
-  **分类输出模块（3个全连接层）**：先通过自适应平均池化将卷积输出的特征图固定为7×7，再通过flatten展平为一维向量。前两层全连接层均为4096维输出，搭配ReLU激活与Dropout（随机丢弃率0.5）抑制过拟合；最后一层全连接层为输出层，维度匹配分类任务的类别数（原论文ImageNet任务为1000维），输出各类别的预测得分。

<img width="1094" height="629" alt="image" src="https://github.com/user-attachments/assets/d3ba607f-30aa-4edc-b7db-8f604a80baff" />
<img width="636" height="545" alt="image" src="https://github.com/user-attachments/assets/99679924-39bb-4ffe-803c-3c4c233de148" />


**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要是去掉后2个卷积块，精简全连接层维度），但核心架构（3×3卷积堆叠+通道数翻倍+ReLU+Dropout）完全保留。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
VGG, a classic deep convolutional neural network proposed in 2014 by Karen Simonyan and Andrew Zisserman from the Visual Geometry Group (VGG) at the University of Oxford, with its findings published in "Very Deep Convolutional Networks for Large-Scale Image Recognition". It achieved a remarkable result in the ImageNet large-scale image classification competition that year, reducing the Top-5 error rate from 15.3% of AlexNet to 7.3%. Its core architecture systematically proved for the first time that **increasing network depth is the key factor to improve the performance of visual tasks**, and established the standard CNN design paradigm of "uniformly using 3×3 small convolution kernels for stacking + doubling the number of channels layer by layer + max pooling for dimensionality reduction". The design idea of VGG is extremely concise and highly scalable. It has not only become the basic architecture template for all subsequent deep convolutional neural networks, but also been widely applied to almost all computer vision fields such as object detection, semantic segmentation, and image generation. It is one of the most influential models in the history of deep learning.

## Architecture
The core architecture of VGG is an **end-to-end deep convolutional neural network with "modular convolution block stacking"**, which is divided into two main parts: a "convolutional feature extraction module" and a "fully connected classification module". The original paper proposed 6 network variants with different depths, among which **VGG16 (13 convolutional layers + 3 fully connected layers)** and **VGG19 (16 convolutional layers + 3 fully connected layers)** are the two most commonly used versions. The original paper's standard input was a 224×224 resolution 3-channel RGB image, and the final output was the predicted probability of the corresponding classification category. The specific structure and design are as follows:

- **Feature Extraction Module (Convolutional Layers)**: It consists of 5 consecutive convolution blocks. Each convolution block stacks 2-4 3×3 convolution kernels (stride 1, padding 1), and all convolutional layers are followed by a ReLU nonlinear activation function. At the end of each convolution block, a 2×2 max pooling layer with stride 2 is connected to halve the size of the feature map. The number of channels starts from 64 in the first convolution block and doubles after each pooling layer, finally reaching 512. This design makes the receptive field of two 3×3 convolutions equivalent to one 5×5 convolution, and the receptive field of three 3×3 convolutions equivalent to one 7×7 convolution, while greatly reducing the number of parameters and increasing the nonlinear expression ability of the network.

- **Classification Output Module (3 Fully Connected Layers)**: First, the feature map output by the convolution is fixed to 7×7 through adaptive average pooling, and then flattened into a one-dimensional vector through flatten. The first two fully connected layers are both 4096-dimensional outputs, combined with ReLU activation and Dropout (random drop rate of 0.5) to suppress overfitting. The last fully connected layer is the output layer, and the dimension matches the number of categories in the classification task (1000 dimensions for the ImageNet task in the original paper), outputting the prediction scores for each category.

<img width="1094" height="629" alt="image" src="https://github.com/user-attachments/assets/1493590c-85d8-4a32-b7da-eaf20eaf5b38" />
<img width="1145" height="540" alt="image" src="https://github.com/user-attachments/assets/7b733b79-6a51-4f62-adab-6a776851bb7b" />


**Note:** We use the CIFAR-10 dataset, which is a 10-class dataset. Unlike the original paper, the image size of CIFAR-10 (32×32) is much smaller than the 224×224 in the original paper. We will make minor adaptations to the network structure (mainly removing the last 2 convolution blocks and simplifying the dimension of the fully connected layers), but the core architecture (3×3 convolution stacking + doubling the number of channels + ReLU + Dropout) will be completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Hinton's students Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
