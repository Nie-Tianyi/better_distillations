import torch
from torch import nn, optim
from tqdm import tqdm

from utils.base_model import DistillationLoss
from utils.cifar100 import CIFAR100Data


class TinyCNN(nn.Module):
    """极简CNN，参数极少"""

    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()

        self.features = nn.Sequential(
            # 32x32x3 -> 16x16x16
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16x16 -> 8x8x32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8x32 -> 4x4x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)

        if return_features:
            return x, feature
        return x


def fast_train_model(
    model,
    model_name,
    teacher_model=None,
    use_distillation=False,
    epochs=30,
    alpha=0.7,
    temperature=4,
):
    """快速训练模型（目标：10分钟内）"""
    print(f"🚀 开始快速训练 {model_name}...")

    data_manager = CIFAR100Data(batch_size=2048)  # 较小的batch_size
    trainloader, testloader = data_manager.get_dataloaders()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if teacher_model:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    # 优化器 - 使用较大的学习率快速收敛
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 学习率调度 - 余弦退火快速收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 损失函数
    if use_distillation and teacher_model:
        criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练记录
    best_acc = 0

    # 快速训练循环
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_distillation and teacher_model:
                # 知识蒸馏训练
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                student_outputs = model(inputs)
                loss = criterion(student_outputs, teacher_outputs, targets)
            else:
                # 普通训练
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = (
                outputs.max(1) if not use_distillation else student_outputs.max(1)
            )
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 更新学习率
        scheduler.step()

        # 快速测试（每5个epoch测试一次以节省时间）
        if epoch % 5 == 0 or epoch == epochs - 1:
            test_acc = fast_test_model(model, testloader, device)
            print(
                f"Epoch {epoch + 1}/{epochs}: Loss = {running_loss / len(trainloader):.3f}, "
                f"Train Acc = {100.0 * correct / total:.2f}%, Test Acc = {test_acc:.2f}%"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                if use_distillation:
                    torch.save(
                        model.state_dict(),
                        f"../model_weights/{model_name}_distilled_best_{alpha}_{temperature}.pth",
                    )
                else:
                    torch.save(
                        model.state_dict(), f"../model_weights/{model_name}_best.pth"
                    )

    print(
        f"🎉 {model_name} 训练完成! 最佳准确率: {best_acc:.2f}%, Alpha={alpha}, Temperature={temperature}"
    )

    return model, best_acc


def fast_test_model(model, testloader, device, num_batches=20):
    """快速测试（只测试部分数据以节省时间）"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            if i >= num_batches:  # 只测试前20个batch
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total
