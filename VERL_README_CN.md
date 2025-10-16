<h1 style="text-align: center;">veRL: Volcano Engine Reinforcement Learning for LLM</h1>

veRL 是一个为大型语言模型（LLMs）设计的灵活、高效且生产就绪的强化学习训练框架。

veRL 是 **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** 论文的开源版本。

veRL 灵活易用，具有以下特点：

- **轻松扩展多样化的强化学习算法**：混合编程模型结合了单控制器和多控制器范式的优势，使复杂的后训练数据流能够灵活表示和高效执行。允许用户用几行代码构建强化学习数据流。

- **通过模块化API无缝集成现有LLM基础设施**：解耦计算和数据依赖，能够与现有LLM框架无缝集成，如 PyTorch FSDP、Megatron-LM 和 vLLM。此外，用户可以轻松扩展到其他LLM训练和推理框架。

- **灵活的设备映射**：支持将模型灵活部署到不同的GPU集合，以实现高效的资源利用和不同集群规模的可扩展性。

- 与流行的 HuggingFace 模型轻松集成


veRL 具有以下高效特点：

- **最先进的吞吐量**：通过无缝集成现有的最先进LLM训练和推理框架，veRL 实现了高生成和训练吞吐量。

- **通过 3D-HybridEngine 实现高效的 actor 模型重新分片**：消除了内存冗余，显著减少了训练和生成阶段转换期间的通信开销。

<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>文档</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>论文</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>微信</b></a> |

<!-- <a href=""><b>幻灯片</b></a> | -->
</p>

## 新闻

- [2024/12] 团队在 NeurIPS 2024 上展示了 <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a>。[幻灯片](https://github.com/eric-haibin-lin/verl-data/tree/neurips) 和 [视频](https://neurips.cc/Expo/Conferences/2024/workshop/100677) 可用。
- [2024/10] veRL 在 Ray Summit 上展示。[YouTube 视频](https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37) 可用。
- [2024/08] HybridFlow (verl) 被 EuroSys 2025 接收。

## 主要特性

- **FSDP** 和 **Megatron-LM** 用于训练。
- **vLLM** 和 **TGI** 用于推理生成，**SGLang** 支持即将推出。
- huggingface 模型支持
- 监督微调
- 奖励模型训练
- 使用 PPO 进行人类反馈强化学习
- flash-attention 集成，序列打包
- 扩展到 70B 模型和数百个 GPU
- 使用 wandb 和 mlflow 进行实验跟踪


## 入门指南

查看这个 [Jupyter Notebook](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer/verl_getting_started.ipynb) 开始使用单个 24GB L4 GPU 进行 PPO 训练（由 [Lighting Studio](https://lightning.ai/hlin-verl/studios/verl-getting-started) 提供 **免费** GPU 配额）！

**快速入门：**
- [安装](https://verl.readthedocs.io/en/latest/start/install.html)
- [快速入门](https://verl.readthedocs.io/en/latest/start/quickstart.html)

**逐步运行 PPO 示例：**
- 数据和奖励准备
  - [为后训练准备数据（Parquet）](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [为数据集实现奖励函数](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- 理解 PPO 示例
  - [PPO 示例架构](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [配置说明](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [运行 GSM8K 示例](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**可重现的算法基线：**
- [PPO](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**代码解释和高级用法（扩展）：**
- PPO 训练器和工作器
  - [PPO Ray 训练器](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP 后端](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM 后端](https://verl.readthedocs.io/en/latest/index.html)
- 高级用法和扩展
  - [Ray API 设计教程](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [扩展到其他 RL(HF) 算法](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [使用 FSDP 后端添加模型](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [使用 Megatron-LM 后端添加模型](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)


## 引用和致谢

如果您觉得这个项目有帮助，请引用：
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

veRL 的设计灵感来自 Nemo-Aligner、Deepspeed-chat 和 OpenRLHF。该项目被 Anyscale、Bytedance、LMSys.org、Shanghai AI Lab、Tsinghua University、UC Berkeley、UCLA、UIUC 和 University of Hong Kong 采纳和支持。

## 使用 veRL 的出版物
- [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://arxiv.org/abs/2410.21236)
- [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME/)

我们正在招聘！如果您对 MLSys/LLM 推理/多模态对齐的实习/全职机会感兴趣，请给我们发送[邮件](mailto:haibin.lin@bytedance.com)。