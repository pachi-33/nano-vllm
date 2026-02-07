# 开发计划（基于ray的nano-vllm改造系项目废案）

## 项目背景
原计划为 nano-vllm 项目添加基于 Ray 的分布式推理能力，支持跨节点的模型并行和数据并行。现作为废案保存。

## 架构设计概述

### Ray Actor 结构设计
1. **ControllerActor** - 中央控制器，负责全局调度和状态管理
2. **WorkerActor** - 工作节点，负责模型推理，内含现有张量并行
3. **SchedulerActor** - 分层调度器（全局+本地），负责请求队列管理
4. **KVCacheManagerActor** - KV缓存管理器，负责跨节点缓存同步
5. **MetadataStoreActor** - 元数据存储，维护全局状态

### 关键技术点
1. **零拷贝传输** - 使用 Ray Plasma 共享内存避免序列化
2. **混合并行** - 节点内 NCCL 张量并行 + 节点间 Ray 通信
3. **分层调度** - 全局节点选择 + 本地批处理优化
4. **容错恢复** - 心跳检测 + 状态检查点 + 快速重建
5. **通信优化** - KV缓存压缩 + 计算通信重叠

### 性能优化策略
- 动态批处理考虑序列长度相似性
- 动态扩缩容基于负载自动调整
- 异步执行重叠 Prefill/Decode 与 KV 缓存传输
- 分布式前缀缓存共享减少重复计算

## 弃置原因
用户选择保存为废案，暂不进行开发。

## 文件结构建议
```
nanovllm/distributed/ray/
├── __init__.py
├── actors.py          # Ray Actor 实现
├── scheduler.py       # 分层调度器
├── state.py          # 状态管理
├── protocol.py       # 通信协议
├── model_loader.py   # 分布式模型加载
├── optimization.py   # 性能优化
├── fault_tolerance.py # 容错机制
└── metrics.py        # 指标收集
```

## 实施要点备忘
1. 优先实现核心 ControllerActor 和 WorkerActor
2. 保持与现有单进程架构的兼容性
3. 确保 Ray 序列化与 PyTorch 张量的高效处理
4. 动态批处理要考虑现有连续批处理逻辑
5. 容错机制需要状态一致性保证

## 后续考虑
如未来重启此项目，建议：
1. 先实现最小可用版本（仅 Controller + Worker）
2. 重点验证 Ray 与现有张量并行的集成效果
3. 性能基准测试对比单节点 NCCL 实现
4. 逐步添加高级功能（容错、动态扩缩容等）

---
创建时间：2026-01-23
状态：已归档（废案）