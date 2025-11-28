from torch import nn
import torch.nn.functional as F


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