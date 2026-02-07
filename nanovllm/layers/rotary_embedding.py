"""
旋转位置编码层（RoPE - Rotary Position Embedding）

RoPE是一种高效的位置编码方法，通过旋转矩阵将位置信息编码到注意力机制中。
相比传统的绝对位置编码，RoPE具有以下优势：
1. 更好地处理变长序列
2. 在自然语言任务中表现更好
3. 支持相对位置建模，即注意力只依赖于两个token的相对距离
4. 计算效率高，适合长序列处理

旋转编码的基本思想：
对于嵌入向量x，将其分解为两部分[x1, x2]，然后应用二维旋转矩阵：
RoPE(x, θ) = [x1*cos(θ) - x2*sin(θ), x1*sin(θ) + x2*cos(θ)]

其中θ与位置相关，通常是位置m和维度i的函数：
θ(m, i) = m * base^(-2i/d)
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量

    将输入张量x沿最后一个维度分成两半，分别作为复数表示的实部和虚部，
    然后应用旋转矩阵乘以cos和sin值。

    Args:
        x: 输入张量，形状为[..., dim]，其中dim必须是偶数
        cos: 余弦值张量，与旋转角度相关
        sin: 正弦值张量，与旋转角度相关

    Returns:
        应用旋转编码后的张量，保持原始形状和数据类型

    数学原理:
        对于复数表示 z = x1 + i*x2
        旋转 θ 角度后：e^(i*θ)*z = (x1*cos(θ) - x2*sin(θ)) + i*(x1*sin(θ) + x2*cos(θ))
    """
    # 将输入张量沿最后一个维度分成两部分
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转矩阵变换
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接结果并转换回原始数据类型
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    该模块为注意力机制中的query和key向量添加旋转位置信息。
    主要特点：
    1. 预计算所有位置的cos和sin值，存储为缓存
    2. 支持任意位置向量的快速旋转编码
    3. 使用torch.compile优化性能

    Args:
        head_size: 注意力头的维度大小
        rotary_dim: 应用旋转编码的维度，通常等于head_size
        max_position_embeddings: 支持的最大序列长度
        base: 旋转角度计算的基数，默认值通常为10000

    缓存说明：
        cos_sin_cache: 预计算的cos和sin值，形状为[max_position, 1, cos_sin_dim]
        其中cos_sin_dim = rotary_dim (一半是cos，一半是sin)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 确保旋转维度等于头维度，这是标准实现
        assert rotary_dim == head_size, f"rotary_dim ({rotary_dim}) must equal head_size ({head_size})"

        # 计算频率的倒数，用于旋转角度计算
        # inv_freq[i] = 1 / (base^(2i/d))，其中d是rotary_dim
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # 位置索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # 计算频率：外积得到每个位置、每个维度的频率
        # freqs[i,j] = t[i] * inv_freq[j]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 计算cos和sin值
        cos = freqs.cos()
        sin = freqs.sin()

        # 合并cos和sin缓存，并添加维度以便广播
        # 最终形状: [max_position, 1, rotary_dim]
        # 其中前一半是cos值，后一半是sin值
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # 注册为buffer，不会作为模型参数更新
        # persistent=False表示不会保存到state_dict
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，为query和key添加旋转位置编码

        Args:
            positions: token的位置索引，形状为[batch, seq_len] 或 [total_tokens]
            query: query向量，形状为[..., head_size]
            key: key向量，形状为[..., head_size]

        Returns:
            (query, key): 添加了位置编码的query和key向量

        说明：
            1. 从预计算缓存中提取对应位置的cos和sin值
            2. 分别对query和key应用旋转编码
            3. 返回旋转编码后的向量
        """
        # 从缓存中提取对应位置的cos和sin值
        # cos_sin shape: [batch, 1, head_size] 或 [total_tokens, 1, head_size]
        cos_sin = self.cos_sin_cache[positions]

        # 分离cos和sin，各占一半维度
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 应用旋转编码到query和key
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    工厂函数：获取RotaryEmbedding实例

    使用LRU缓存确保相同参数只创建一个实例，节省内存。
    支持rope_scaling参数扩展（当前未实现）。

    Args:
        head_size: 注意力头维度大小
        rotary_dim: 旋转维度
        max_position: 最大位置编码长度
        base: 基数
        rope_scaling: 旋转编码缩放配置（未使用）

    Returns:
        RotaryEmbedding: 旋转位置编码模块实例
    """
    assert rope_scaling is None, "rope_scaling尚未实现"
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
