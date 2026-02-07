"""
归一化层模块

该模块实现了高效的RMSNorm (Root Mean Square Layer Normalization) 归一化方法。
相比传统Layer Norm，RMSNorm只进行缩放而不进行平移，计算更简单且性能相当。
主要应用于现代大语言模型中，如LLaMA、Qwen等。

RMSNorm计算公式：
    RMSNorm(x) = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2))

当提供residual时，该模块还可以同时处理残差连接和归一化。
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMS归一化层

    RMSNorm (Root Mean Square Layer Normalization) 是一种高效的归一化方法。
    相比LayerNorm，RMSNorm只进行缩放而不进行平移（没有bias参数），
    这减少了参数量但保持了性能。

    主要特点：
    1. 无平移参数，只使用缩放权重
    2. 支持残差连接优化
    3. 使用float32计算防止数值溢出
    4. 支持torch.compile编译优化

    Args:
        hidden_size: 隐藏层维度大小
        eps: 数值稳定性的小常数
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # 数值稳定性常数
        self.eps = eps
        # 缩放参数（RMSNorm没有bias，只有weight）
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMS归一化的核心实现

        Args:
            x: 输入张量，任何形状

        Returns:
            归一化后的张量，保持输入形状

        计算步骤：
        1. 保存原始数据类型
        2. 转换为float32防止数值溢出
        3. 计算平方均值
        4. 计算RMS和归一化因子
        5. 应用归一化和缩放
        """
        # 保存原始数据类型
        orig_dtype = x.dtype
        # 转换为float32进行计算，防止数值溢出
        x = x.float()

        # 计算每个token的平方的均值（沿特征维度）
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # 计算归一化因子 rsqrt(var + eps) = 1/sqrt(var + eps)
        # 并应用到x上（原地操作节省内存）
        x.mul_(torch.rsqrt(var + self.eps))

        # 转换回原始数据类型并应用权重缩放
        x = x.to(orig_dtype).mul_(self.weight)

        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        带残差连接的RMS归一化

        该函数同时完成残差相加和归一化处理，减少一次内存读写。

        Args:
            x: 主输入张量
            residual: 残差张量，与x形状相同

        Returns:
            (normalized, residual_pre_norm):
            - normalized: 归一化后的结果（包含归一化权重）
            - residual_pre_norm: 标准化前的原始残差和（已转换回原始数据类型）

        计算步骤：
        1. 残差相加（在float32精度下）
        2. 保存标准化前的结果（转换为原始数据类型）
        3. 在float32精度下计算RMS
        4. 应用归一化和权重
        """
        # 保存原始数据类型
        orig_dtype = x.dtype

        # 残差相加（在float32精度下避免溢出）
        x = x.float().add_(residual.float())

        # 保存残差和（用于返回）
        residual = x.to(orig_dtype)

        # 计算RMS
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # 应用归一化和权重
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数

        Args:
            x: 输入张量
            residual: 可选的残差张量
                     - None: 只进行归一化
                     - Tensor: 同时进行残差相加和归一化

        Returns:
            如果residual为None：返回归一化后的张量
            如果residual有值：返回(normalized, residual_before_norm)元组
                          - normalized是归一化结果
                          - residual_before_norm是相加但尚未归一化的结果
        """
        if residual is None:
            # 只进行RMS归一化
            return self.rms_forward(x)
        else:
            # 同时进行残差相加和归一化
            return self.add_rms_forward(x, residual)
