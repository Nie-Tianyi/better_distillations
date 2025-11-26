import torch
import torchvision
import torchvision.transforms as transforms

def download_cifar100():
    print("开始下载CIFAR-100数据集...")

    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 下载训练集
    print("下载训练集中...")
    trainset = torchvision.datasets.CIFAR100(
        root='.././data',
        train=True,  # 这是训练集
        download=True,
        transform=transform_train
    )

    # 下载测试集
    print("下载测试集中...")
    testset = torchvision.datasets.CIFAR100(
        root='.././data',
        train=False,  # 这是测试集
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    print("下载完成！")
    print(f"训练集样本数: {len(trainset)}")
    print(f"测试集样本数: {len(testset)}")
    print(f"类别数: {len(trainset.classes)}")

    return trainset, testset, trainloader, testloader

# 执行下载
if __name__ == "__main__":
    trainset, testset, trainloader, testloader = download_cifar100()