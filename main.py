import torch
from nbconvert.filters.filter_links import resolve_one_reference

from utils.base_model import count_parameters, MicroResNet
from utils.res_net import ResNet20, ResNet56
from utils.tiny_cnn import TinyCNN


def main():
    print("Hello from quantized-distillation!")
    print("PyTorch version:", torch.version.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)


if __name__ == "__main__":
    # 测试模型大小
    tiny_cnn = TinyCNN(num_classes=100)
    print(f"TinyCNN 参数数量: {count_parameters(tiny_cnn):,}")  # 约 0.31M 参数
    micro_resnet = MicroResNet(num_classes=100)
    print(f"MicroResNet 参数数量: {count_parameters(micro_resnet):,}")  # 约 0.08M 参数
    resnet20 = ResNet20(num_classes=100)
    print(f"ResNet20 参数数量: {count_parameters(resnet20):,}")  # 约 17M 参数
    resnet56 = ResNet56(num_classes=100)
    print(f"ResNet56 参数数量: {count_parameters(resnet56):,}")  # 约 55M 参数
