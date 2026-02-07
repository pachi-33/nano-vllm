"""
激活函数层模块

该模块定义了各种自定义激活函数实现，包括：
1. SwiGLU变种：SiluAndMul

这些激活函数特别为大语言模型设计，结合了高性能和数值稳定性。
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SwiGLU激活函数的实现

    SwiGLU (Swish-Gated Linear Unit) 是一种高效的激活函数，结合了Swish激活函数
    和门控机制。计算公式：
    SwiGLU(x) = Swish(W1*x) * W2*x = Sigmoid(W1*x) * W1*x * W2*x

    为了简化实现，我们假设输入已经被预先分割并投影，因此实现为：
    SiluAndMul(x) = SiLU(x1) * x2
    其中x是拼接的张量 [x1, x2]，沿最后一个维度分割

    该激活函数在LLaMA、PaLM等模型中表现优异。

    参考文章：
    GLU Variants Improve Transformer:
    https://arxiv.org/abs/2002.05202
    """

    def __init__(self):
        """初始化SwiGLU模块"""
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状任意，
              最后一个维度必须是偶数，会被平均分割成两部分

        Returns:
            激活后的张量，形状与输入相同
            计算: SiLU(x1) * x2，其中x = [x1, x2]

        示例:
            >>> x = torch.randn(2, 4, 128)  # 最后一个维度是偶数
            >>> output = silu_and_mul(x)    # 输出形状: [2, 4, 128]
            >>> # 内部会将128分成64+64，计算SiLU(x1) * x2
        """
        # 沿最后一个维度将输入分割成两部分
        # x1和y各占一半维度
        x, y = x.chunk(2, -1)

        # 计算SwiGLU: SiLU(x) * y
        # F.silu(x) = x * sigmoid(x)
        return F.silu(x) * y
