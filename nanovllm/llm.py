"""
Nano-VLLM 主入口模块

这个模块提供了一个简单的门面类 LLM，用于用户访问整个推理引擎的功能。
LLM 类继承自 LLMEngine，所有的核心功能都在 LLMEngine 中实现。
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    大语言模型推理门面类

    这是 Nano-VLLM 的主要用户接口类，通过继承 LLMEngine 提供了所有推理功能。
    用户通过实例化这个类来使用模型的推理能力，支持生成文本、批量推理等功能。

    示例:
        >>> from nanovllm import LLM, SamplingParams
        >>> llm = LLM(model="Qwen/Qwen3-8B")
        >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        >>> output = llm.generate("你好，", sampling_params)
    """
    pass
