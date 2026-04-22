import torch
import torch.nn as nn

# ----------------------
# 原论文定义的 MLPConv 模块
# ----------------------
class MLPConv(nn.Module):
    """
    NiN原论文中的微网络卷积模块
    参数 last_relu: 控制最后一个1×1卷积后是否加ReLU（最后一个MLPConv设为False）
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, last_relu=True):
        super(MLPConv, self).__init__()
        
        # 构建层列表：标准卷积 → ReLU → 1×1卷积 → ReLU → 1×1卷积（可选ReLU）
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        ]
        
        # 仅在非最后一个MLPConv时添加最后的ReLU
        if last_relu:
            layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# ----------------------
# 原论文定义的完整 NiN 网络（针对 ImageNet 224×224 输入，1000类输出）
# ----------------------
class NiN_ImageNet(nn.Module):
    """
    严格复现原论文《Network In Network》中针对ImageNet的NiN架构
    输入尺寸：224×224×3 RGB图像
    输出尺寸：1000维分类logits（无Softmax）
    """
    def __init__(self, num_classes=1000):
        super(NiN_ImageNet, self).__init__()
        
        # 原论文完整特征提取堆叠
        self.features = nn.Sequential(
            # 第1个MLPConv块：11×11大卷积核，步长4
            MLPConv(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 重叠最大池化
            
            # 第2个MLPConv块：5×5卷积核
            MLPConv(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 重叠最大池化
            
            # 第3个MLPConv块：3×3卷积核
            MLPConv(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 重叠最大池化
            
            # 第4个MLPConv块：输出通道数=类别数，且最后不加ReLU
            MLPConv(384, num_classes, kernel_size=3, stride=1, padding=1, last_relu=False)
        )
        
        # 原论文核心：全局平均池化替代全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, num_classes)
        return x
    
if __name__ == "__main__":
    # 原论文标准输入：224×224×3，batch_size=2
    model = NiN_ImageNet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"原论文输入尺寸: {x.shape}")    # 输出: torch.Size([2, 3, 224, 224])
    print(f"原论文输出尺寸: {output.shape}") # 输出: torch.Size([2, 1000])