import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import time
from utils.cifar100 import CIFAR100Data
from utils.res_net import ResNet56, ResNet20
import torch.nn.functional as F


def train_teacher_model():
    """训练教师模型 ResNet-56"""
    print("🚀 开始训练教师模型 ResNet-56...")

    # 数据加载

    data_manager = CIFAR100Data(batch_size=128)
    trainloader, testloader = data_manager.get_dataloaders()

    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = ResNet56(num_classes=100).to(device)

    optimizer = optim.SGD(
        teacher_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    test_accuracies = []
    best_acc = 0

    # 训练循环
    for epoch in range(200):
        teacher_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = teacher_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        # 学习率调度
        scheduler.step()

        # 测试准确率
        test_acc = test_model(teacher_model, testloader, device)
        test_accuracies.append(test_acc)
        train_losses.append(running_loss / len(trainloader))

        print(f"Epoch {epoch + 1}: Test Accuracy = {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                teacher_model.state_dict(), "../model_weights/teacher_resnet56_best.pth"
            )
            print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

    # 保存最终模型
    torch.save(
        teacher_model.state_dict(), "../model_weights/teacher_resnet56_final.pth"
    )
    print(f"🎉 教师模型训练完成! 最佳准确率: {best_acc:.2f}%")

    return teacher_model, train_losses, test_accuracies


def test_model(model, testloader, device):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def train_student_vanilla():
    """训练学生模型 ResNet-20（无知识蒸馏）"""
    print("🚀 开始训练学生模型 ResNet-20（无蒸馏）...")

    data_manager = CIFAR100Data(batch_size=128)
    trainloader, testloader = data_manager.get_dataloaders()

    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = ResNet20(num_classes=100).to(device)

    optimizer = optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    best_acc = 0

    # 训练循环
    for epoch in range(200):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student Vanilla Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        scheduler.step()

        # 测试准确率
        test_acc = test_model(student_model, testloader, device)
        print(f"Student Vanilla Epoch {epoch + 1}: Test Accuracy = {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                student_model.state_dict(),
                "../model_weights/student_vanilla_resnet20_best.pth",
            )
            print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

    torch.save(
        student_model.state_dict(),
        "../model_weights/student_vanilla_resnet20_final.pth",
    )
    print(f"🎉 学生模型（无蒸馏）训练完成! 最佳准确率: {best_acc:.2f}%")

    return student_model, best_acc


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, alpha=0.7, temperature=4):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # 知识蒸馏损失（软标签）
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
        ) * (self.temperature**2)

        # 交叉熵损失（硬标签）
        hard_loss = self.ce_loss(student_logits, targets)

        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train_student_with_distillation(teacher_model):
    """使用知识蒸馏训练学生模型"""
    print("🚀 开始训练学生模型 ResNet-20（带知识蒸馏）...")

    data_manager = CIFAR100Data(batch_size=128)
    trainloader, testloader = data_manager.get_dataloaders()

    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = ResNet20(num_classes=100).to(device)

    # 加载预训练的教师模型
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = DistillationLoss(alpha=0.7, temperature=4)

    # 训练记录
    best_acc = 0

    # 训练循环
    for epoch in range(200):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student KD Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 学生模型输出
            student_outputs = student_model(inputs)

            # 教师模型输出（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # 计算蒸馏损失
            loss = criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        scheduler.step()

        # 测试准确率
        test_acc = test_model(student_model, testloader, device)
        print(f"Student KD Epoch {epoch + 1}: Test Accuracy = {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                student_model.state_dict(),
                "../model_weights/student_kd_resnet20_best.pth",
            )
            print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

    torch.save(
        student_model.state_dict(), "../model_weights/student_kd_resnet20_final.pth"
    )
    print(f"🎉 学生模型（知识蒸馏）训练完成! 最佳准确率: {best_acc:.2f}%")

    return student_model, best_acc


class MicroResNet(nn.Module):
    """超小型ResNet，适合快速实验"""

    def __init__(self, num_classes=100):
        super(MicroResNet, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 残差块
        self.block1 = self._make_block(16, 16, stride=1)
        self.block2 = self._make_block(16, 32, stride=2)
        self.block3 = self._make_block(32, 64, stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_block(self, in_channels, out_channels, stride):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)

        if return_features:
            return x, feature
        return x


def train_micro_model():
    """训练超小型ResNet模型"""
    # 数据加载
    data_manager = CIFAR100Data(batch_size=128)
    trainloader, testloader = data_manager.get_dataloaders()

    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroResNet(num_classes=100).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    best_acc = 0
    # 训练循环
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"MicroResNet Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )
        scheduler.step()

        test_acc = test_model(model, testloader, device)
        print(f"MicroResNet Epoch {epoch + 1}: Test Accuracy = {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                model.state_dict(),
                "../model_weights/micro_resnet20_best.pth",
            )
            print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

    torch.save(model.state_dict(), "../model_weights/micro_resnet20_final.pth")
    print(f"🎉 超小型ResNet训练完成! 最佳准确率: {best_acc:.2f}%")

    return model, best_acc


def train_micro_student_with_distillation(teacher_model, alpha=0.7, temperature=4):
    """使用知识蒸馏训练学生模型"""
    print("🚀 开始训练学生模型 MicroResNet（带知识蒸馏）...")

    data_manager = CIFAR100Data(batch_size=128)
    trainloader, testloader = data_manager.get_dataloaders()

    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = MicroResNet(num_classes=100).to(device)

    # 加载预训练的教师模型
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)

    # 训练记录
    best_acc = 0

    # 训练循环
    for epoch in range(200):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student KD Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 学生模型输出
            student_outputs = student_model(inputs)

            # 教师模型输出（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # 计算蒸馏损失
            loss = criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        scheduler.step()

        # 测试准确率
        test_acc = test_model(student_model, testloader, device)
        print(f"Student KD Epoch {epoch + 1}: Test Accuracy = {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                student_model.state_dict(),
                "../model_weights/student_kd_micro_resnet20_best.pth",
            )
            print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

    torch.save(
        student_model.state_dict(), "../model_weights/student_kd_micro_resnet20_final.pth"
    )
    print(f"🎉 MicroResNet（知识蒸馏）训练完成! 最佳准确率: {best_acc:.2f}%")

    return student_model, best_acc


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型大小
    micro_model = MicroResNet(num_classes=100)
    print(f"MicroResNet 参数数量: {count_parameters(micro_model):,}")  # 约 0.1M 参数
