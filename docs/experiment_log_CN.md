## 实验日志

### 初步结果

资源：[wandb](https://wandb.ai/peterjin/Search-R1-open)


初步实验仅在自然问题（NQ）数据集（+ PPO）上进行，使用少量训练步骤。


### v0.1

资源：[wandb](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.1)


我们将实验从 NQ 扩展到七个数据集，使用 PPO 和 GRPO 方法。研究仍在少量训练步骤和大学习率预热比下进行。


### v0.2

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.2)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.2)、[论文](https://arxiv.org/abs/2503.09516)


我们修复了几个错误，包括[检索到的标记掩码](https://github.com/PeterGriffinJin/Search-R1/pull/21)和[GRPO 样本索引](https://github.com/PeterGriffinJin/Search-R1/commit/9ec2fa9892fbf0315d0c67b4dc08ae8f6cf5f378)。
前者可以大大提高强化学习训练的稳定性。
然后我们调整训练脚本，增加训练步骤数量并降低学习率预热比，以获得更好的性能，并在不同规模的 LLM（3B、7B、14B）上进行实验。


### v0.3

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.3)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.3)、[论文](https://arxiv.org/abs/2505.15117)

我们对以下方面进行了研究：(1) 奖励设计；(2) LLM 主干；和 (3) 搜索引擎。

- 奖励设计
  - 格式奖励
  - 中间检索奖励
- LLM 主干
  - LLM 类型（例如，通用 LLM 或推理 LLM）
  - LLM 规模（3B/7B/14B/32B）
- 搜索引擎
  - 强化学习训练动态
  - 推理过程中的泛化
- 数据扩展

详细信息可以在[论文](https://arxiv.org/abs/2505.15117)中找到。