import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------
# 1. NiN 模型定义
# ----------------------
class MLPConv(nn.Module):
    """NiN的基本微网络卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, last_relu=True):
        super(MLPConv, self).__init__()
        # 构建层列表，最后一层 ReLU 可选
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        ]
        if last_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class NiN_CIFAR10(nn.Module):
    """适配 CIFAR-10 的 NiN 网络（修复版）"""
    def __init__(self):
        super(NiN_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            MLPConv(3, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            MLPConv(192, 160, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            MLPConv(160, 96, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 最后一个 MLPConv 不加 ReLU
            MLPConv(96, 10, kernel_size=3, stride=1, padding=1, last_relu=False)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加 Kaiming 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        """Kaiming 初始化，适合 ReLU 激活函数的网络"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ----------------------
# 2. 数据加载与预处理
# ----------------------
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ----------------------
# 3. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    lr = 0.01  
    num_epochs = 20

    model = NiN_CIFAR10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # 添加学习率衰减，进一步提升后期性能
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_loader, test_loader = get_data_loaders(batch_size)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        scheduler.step()
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), 'nin_cifar10_fixed.pth')
    print("Model saved as nin_cifar10_fixed.pth")

    # ----------------------
    # 可视化绘图
    # ----------------------
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 7))

    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')

    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, num_epochs + 1, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('NiN Training Metrics (CIFAR-10, Fixed)', fontsize=16)

    plt.savefig('nin_training_curve_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()