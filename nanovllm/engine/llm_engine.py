"""
LLM 推理引擎核心模块

该模块提供了 LLMEngine 类，是整个 nano-vllm 框架的核心协调器。
它负责管理推理流程、调度请求、协调多进程计算、处理输入输出等。

主要功能：
1. 初始化模型和多进程环境
2. 接受用户请求并调度
3. 执行模型推理步骤
4. 管理文本生成流程
5. 统计性能指标（吞吐量等）

使用示例：
    >>> config = Config(model='Qwen/Qwen3-8B')
    >>> engine = LLMEngine(config)
    >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    >>> engine.add_request('你好，', sampling_params)
    >>> outputs, num_tokens = engine.step()
    >>> results = engine.generate(['你好，'], sampling_params)
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎核心类

    负责协调整个文本生成流程，包括：
    - 初始化模型和环境
    - 管理多进程并行计算
    - 调度用户请求
    - 执行推理步骤
    - 收集和返回结果

    该类作为框架的核心控制器，连接了各个子模块（调度器、模型运行器、序列管理等）。
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLM 推理引擎

        该方法会配置并启动整个推理环境，包括：
        1. 解析并创建配置对象
        2. 初始化多进程环境（如果需要张量并行）
        3. 在主进程和子进程中创建模型运行器
        4. 加载分词器
        5. 初始化调度器
        6. 注册退出时的清理函数

        Args:
            model: 模型名称或路径，例如 "Qwen/Qwen3-8B"
            **kwargs: 配置参数，例如 tensor_parallel_size, max_model_len 等

        Note:
            如果 tensor_parallel_size > 1，会启动多个进程来进行张量并行计算，
            每个进程会在一个独立的 GPU 上运行部分模型。
        """
        # 从 kwargs 中筛选出 Config 支持的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 初始化进程列表和事件列表，用于张量并行管理
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 启动子进程进行张量并行计算
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 在主进程中创建模型运行器
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载分词器并配置结束符
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 创建调度器管理请求队列
        self.scheduler = Scheduler(config)

        # 注册退出时的清理函数
        atexit.register(self.exit)

    def exit(self):
        """
        清理并退出推理引擎

        该方法会：
        1. 通知所有进程退出
        2. 删除模型运行器实例
        3. 等待所有子进程结束

        该方法会在程序退出时自动调用（通过 atexit 注册）。
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个推理请求到调度队列

        该方法将用户输入（文本或 token 列表）和一个请求实例，并将其添加到调度器的队列中。

        Args:
            prompt: 用户输入，可以是字符串或 token ID 列表
            sampling_params: 采样参数，控制生成行为（温度、top_p、max_tokens 等）

        Process:
            1. 如果 prompt 是字符串，使用分词器将其编码为 token ID
            2. 创建 Sequence 对象包装请求
            3. 将序列添加到调度器的等待队列

        Example:
            >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
            >>> engine.add_request("你好，", sampling_params)
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行单步推理

        这是推理引擎的核心执行方法，完成一次完整的推理步骤：
        1. 调度器选择要执行的序列（预填充或解码阶段）
        2. 调用模型运行器执行前向计算
        3. 调度器处理计算结果并更新序列状态

        Returns:
            tuple: (outputs, num_tokens)
                - outputs: 列表，包含已完成的序列（(seq_id, token_ids) 元组）
                - num_tokens: 处理的 token 数（正数表示预填充，负数表示解码）

        Process:
            1. 调度器从等待队列中选择序列，决定是预填充还是解码阶段
            2. 调用模型运行器在选中的序列上执行一步推理
            3. 将生成的 token 添加到序列中，检查是否有序列完成
            4. 返回完成的序列和处理的 token 数量（用于吞吐量计算）

        Note:
            返回值中的 num_tokens 符号有意义：
            - 正数：预填充阶段处理的 token 数
            - 负数：解码阶段生成的序列数（用于吞吐量计算）
        """
        # 调度器选择要执行的序列，并确定是预填充还是解码阶段
        seqs, is_prefill = self.scheduler.schedule()

        # 调用模型运行器执行实际的前向计算
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 调度器后处理：更新序列状态，处理完成序列
        self.scheduler.postprocess(seqs, token_ids)

        # 收集已完成的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 计算处理的 token 数量（用于吞吐量统计）
        # 预填充阶段返回处理的 token 总数，解码阶段返回序列数的负值
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已处理完成

        Returns:
            bool: 如果所有请求都已处理完成返回 True，否则返回 False

        Note:
            该方法会查询调度器，检查是否还有未完成的序列需要处理。
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        生成文本的主接口方法

        该方法提供了完整的文本生成流程，包括：
        1. 添加所有请求到调度队列
        2. 循环执行推理直到所有请求完成
        3. 实时显示进度和性能指标（吞吐量）
        4. 收集并返回所有生成的文本

        Args:
            prompts: 提示列表，可以是字符串列表或 token ID 列表的列表
            sampling_params: 采样参数，可以是单个对象或列表（与 prompts 对应）
            use_tqdm: 是否显示进度条，默认为 True

        Returns:
            list[dict]: 生成结果列表，每个元素是包含以下字段的字典：
                - "text": 生成的文本字符串
                - "token_ids": 生成的 token ID 列表

        Process:
            1. 创建进度条（如果启用）
            2. 如果采样参数是单个对象，复制到与 prompts 相同长度
            3. 将所有请求添加到调度队列
            4. 循环执行：
                a. 记录时间戳
                b. 执行单步推理
                c. 计算并更新吞吐率（预填充和解码阶段分开统计）
                d. 收集已完成的序列
                e. 更新进度条
            5. 将所有完成的序列按顺序排列
            6. 使用分词器将 token ID 解码为文本
            7. 返回结果列表

        Example:
            >>> from nanovllm import SamplingParams
            >>> llm = LLMEngine(model="Qwen/Qwen3-8B")
            >>> sp = SamplingParams(temperature=0.7, max_tokens=100)
            >>> prompts = ["你好，", "今天天气"]
            >>> results = llm.generate(prompts, sp)
            >>> print(results[0]["text"])

        Performance Metrics:
            进度条会实时显示两个关键性能指标：
            - Prefill: 预填充阶段的吞吐率（token/s）
            - Decode: 解码阶段的吞吐率（token/s）
        """
        # 创建进度条用于显示生成进度和性能
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 如果采样参数不是列表，复制多个以匹配所有 prompts
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将所有请求添加到调度队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 初始化输出字典和吞吐率统计
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 循环执行推理直到所有请求完成
        while not self.is_finished():
            # 记录推理开始时间，用于计算吞吐率
            t = perf_counter()

            # 执行单步推理，获取输出和 token 数量
            output, num_tokens = self.step()

            # 更新进度条和吞吐率统计
            if use_tqdm:
                # 根据 num_tokens 判断是预填充阶段还是解码阶段
                if num_tokens > 0:  # 预填充阶段
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:  # 解码阶段
                    decode_throughput = -num_tokens / (perf_counter() - t)

                # 更新进度条后显示（包含预填充和解码吞吐率）
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 处理已完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条

        # 按照序列 ID 排序输出结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 将 token ID 解码为文本，并整理成最终输出格式
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()

        return outputs
