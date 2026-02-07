"""
序列管理模块 - 推理请求的核心数据结构

该模块定义了Sequence类和SequenceStatus枚举，管理文本推理过程中的序列状态。
每个请求被封装为一个Sequence对象，保存token历史、生成状态和KV缓存位置。

核心概念：
1. **序列状态转换**：
   WAITING（等待） → RUNNING（运行） → FINISHED（完成）
   - WAITING：新请求，等待调度器分配
   - RUNNING：正在被处理（预填充或解码）
   - FINISHED：生成完成或达到长度限制

2. **分块管理**：
   - 序列按block_size切分为块
   - 每个块对应KV缓存中的一段连续空间
   - block_table记录块到缓存位置的映射

3. **缓存优化**：
   - num_cached_tokens：已存储到KV缓存的token数
   - 避免重复存储和计算已缓存的token

4. **序列属性**：
   - token_ids：完整的token历史（prompt + completion）
   - num_prompt_tokens：提示部分长度（固定）
   - completion_token_ids：生成的token（动态增长）
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举

    定义了序列在推理过程中的三种状态：
    - WAITING: 等待调度（新请求）
    - RUNNING: 运行中（已分配资源）
    - FINISHED: 已完成（生成结束或达到限制）
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类 - 表示一个推理请求的生命周期

    管理从token历史到KV缓存位置的所有信息，是推理过程的核心数据结构。

    主要属性：
    - seq_id: 唯一序列ID（递增）
    - status: 序列状态（WAITING/RUNNING/FINISHED）
    - token_ids: 完整的token历史（prompt + completion）
    - last_token: 最新的token（用于快速访问）
    - num_tokens: 总token数
    - num_prompt_tokens: 提示长度（固定）
    - num_cached_tokens: 已缓存到KV缓存的token数
    - block_table: 块表（映射到KV缓存位置）
    - temperature/max_tokens/ignore_eos: 采样参数

    块管理：
    - block_size: 块大小（类属性，固定256 token）
    - 序列按块划分，便于内存管理和前缀缓存
    - block()方法获取第i个块的token

    生成过程：
    - 初始状态：WAITING，只有prompt_tokens
    - 预填充后：token_ids包含所有token，num_cached_tokens增长
    - 解码中：每次append_token()添加新token
    - 完成：状态变为FINISHED
    """

    block_size = 256  # 类属性：块大小（256 token）
    counter = count()  # 全局序列ID生成器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列

        Args:
            token_ids: 提示token列表（初始序列）
            sampling_params: 采样参数（温度、最大长度等）
        """
        self.seq_id = next(Sequence.counter)  # 分配唯一ID
        self.status = SequenceStatus.WAITING  # 初始状态为等待
        self.token_ids = copy(token_ids)  # 复制token列表（避免外部修改）
        self.last_token = token_ids[-1]  # 保存最后一个token（快速访问）
        self.num_tokens = len(self.token_ids)  # 总token数
        self.num_prompt_tokens = len(token_ids)  # 提示长度（固定）
        self.num_cached_tokens = 0  # 已缓存的token数（初始为0）
        self.block_table = []  # 块表（映射到KV缓存）
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大生成长度
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略结束符

    def __len__(self):
        """返回序列长度（token数）"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引访问：seq[i]返回第i个token"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """检查序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的completion token数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """提示部分的token列表"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """生成的completion token列表"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        已完全缓存的块数

        等于num_cached_tokens除以块大小（向下取整）
        表示有多少个完整的块已经完全存储到KV缓存
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        序列占用的总块数（包括不完整的块）

        计算公式：ceil(num_tokens / block_size)
        表示序列需要多少个KV缓存块
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块的token数（可能不完整）

        取值范围：1到block_size
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第i个块的token列表

        Args:
            i: 块索引（从0开始）

        Returns:
            list[int]: 第i个块的token列表

        Raises:
            AssertionError: 如果索引超出范围
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        追加一个新生成的token

        Args:
            token_id: 新生成的token ID

        更新：
        - token_ids列表
        - last_token（最新token）
        - num_tokens计数
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化（用于pickle或共享内存）

        优化存储：
        - 只保留必要的状态信息
        - 如果completion为空，只保存prompt部分
        - 否则只保存最后一个token（解码中只关心最新token）

        Returns:
            tuple: 序列化状态
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化

        Args:
            state: __getstate__返回的序列化状态
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
