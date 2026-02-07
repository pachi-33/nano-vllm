"""
采样器模块

该模块实现了从模型logits生成token的采样逻辑。
使用Gumbel-Softmax技术实现高效的分类采样，
相比传统的multinomial采样有数学等价性但计算更高效。

采样过程：
1. 温度缩放：logits = logits / temperature
2. Softmax转换为概率：probs = softmax(logits)
3. Gumbel噪声采样：使用指数分布生成Gumbel噪声
4. 选择最大概率的token

数学原理：
Gumbel-Softmax采样等价于从分类分布采样
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    文本生成采样器

    使用Gumbel-Softmax技术从模型输出分布中采样生成token。
    支持温度控制来调整生成的随机性。

    关键特性：
    1. 使用torch.compile加速采样过程
    2. 支持批量采样，不同序列可使用不同温度
    3. 数学上等价于从多项分布采样，但更高效
    """

    def __init__(self):
        """初始化采样器"""
        super().__init__()


    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        执行分类采样

        Args:
            logits: 模型的原始输出，形状为[batch_size, vocab_size]
            temperatures: 每个序列的温度参数，形状为[batch_size]
                          控制生成的随机性：
                          - 低温度（如0.1）：更确定性的输出
                          - 高温度（如1.5）：更随机的输出

        Returns:
            sample_tokens: 采样的token索引，形状为[batch_size]

        采样步骤说明：
        1. 温度缩放：将logits除以温度参数
           低温下差异被放大，更容易选中最大概率项
           高温下差异被缩小，采样更随机
        2. Softmax转换：将logits转换为概率分布
        3. Gumbel采样：
           - 生成指数随机数：exponential(1)
           - 转换为Gumbel噪声：-log(-log(U))
           - 添加噪声后除以概率：probs / gumbel_noise
        4. 选择argmax：选择得分最高的类别

        数学等价性：
        这个过程数学上等价于从多项分布Multinomial(probs)中采样，
        但避免了显式地从分类分布中采样，更加高效。
        """
        # 确保使用float32精度进行计算
        # 对每个序列的logits应用温度缩放
        # temperatures.unsqueeze(dim=1) 使其从[batch]变成[batch,1]便于广播
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 将logits转换为概率分布
        # Softmax确保所有概率和为1
        probs = torch.softmax(logits, dim=-1)

        # Gumbel-Softmax采样
        # 从指数分布(λ=1)采样得到随机数，等价于Gumbel噪声的生成步骤
        # exponential_(1) 生成形状与probs相同的张量
        # clamp_min_(1e-10) 避免除零错误
        # probs.div_(...) 实现Gumbel-Softmax的核心操作
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        return sample_tokens
