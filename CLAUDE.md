# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Search-R1 是一个专门用于训练**推理与搜索交错的大型语言模型**的强化学习框架。它基于 veRL（Volcano Engine Reinforcement Learning）框架构建，使语言模型能够学会协调进行推理和工具调用（如调用搜索引擎）。

## 项目文件组织结构

```
Search-R1/
├── 📁 docs/                              # 文档目录
│   ├── experiment_log_CN.md             # 实验日志（中文）
│   ├── experiment_log.md                # 实验日志（英文）
│   ├── multinode_CN.md                  # 多节点训练指南（中文）
│   ├── multinode.md                     # 多节点训练指南（英文）
│   ├── retriever_CN.md                  # 检索器文档（中文）
│   └── retriever.md                     # 检索器文档（英文）
│
├── 📁 example/                           # 示例配置目录
│   ├── 📁 multinode/                     # 多节点训练示例
│   │   ├── train_grpo_multinode_32b.sh  # GRPO多节点32B模型训练脚本
│   │   ├── train_grpo_multinode_72b.sh  # GRPO多节点72B模型训练脚本
│   │   └── train_ppo_multinode_32b.sh   # PPO多节点32B模型训练脚本
│   ├── 📁 retriever/                     # 检索器启动示例
│   │   ├── retrieval_launch_ann.sh      # ANN检索器启动脚本
│   │   ├── retrieval_launch_bm25.sh     # BM25检索器启动脚本
│   │   ├── retrieval_launch_google.sh   # Google搜索启动脚本
│   │   ├── retrieval_launch_hierarchical.sh  # 层次化检索启动脚本
│   │   └── retrieval_launch_serpapi.sh  # SERPAPI搜索启动脚本
│   ├── case.txt                         # 示例用例
│   └── corpus.jsonl                     # 示例语料
│
├── 📁 public/                            # 公共资源目录
│   ├── head.png                         # 项目头部图片
│   ├── llama32-3b.png                   # LLaMA32-3B模型图片
│   ├── logo.png                         # 项目Logo
│   ├── main.png                         # 主图
│   ├── multi-turn.png                   # 多轮交互图
│   ├── single-turn.png                  # 单轮交互图
│   ├── status.png                       # 状态图
│   └── worker.png                       # 工作节点图
│
├── 📁 scripts/                           # 脚本目录
│   ├── 📁 data_process/                  # 数据处理脚本
│   │   ├── nq_rag.py                    # NQ数据集RAG处理
│   │   ├── nq_search.py                 # NQ数据集搜索处理
│   │   ├── nq.py                        # NQ数据集基础处理
│   │   ├── qa_search_test_merge.py      # QA搜索测试集合并
│   │   └── qa_search_train_merge.py     # QA搜索训练集合并
│   ├── 📁 nq_hotpotqa/                   # NQ和HotpotQA数据处理
│   │   ├── 📁 v0.1/                      # 版本0.1配置
│   │   ├── 📁 v0.2/                      # 版本0.2配置
│   │   ├── 📁 v0.3/                      # 版本0.3配置
│   │   ├── data_process.sh              # 数据处理脚本
│   │   ├── evaluate.sh                  # 评估脚本
│   │   ├── README_CN.md                 # 说明文档（中文）
│   │   └── README.md                    # 说明文档（英文）
│   ├── download.py                      # 数据下载脚本
│   ├── download.sh                      # 数据下载Shell脚本
│   ├── upload.py                        # 数据上传脚本
│   └── upload.sh                        # 数据上传Shell脚本
│
├── 📁 search_r1/                         # 核心搜索与推理模块
│   ├── 📁 llm_agent/                     # LLM代理模块
│   │   ├── __init__.py                  # 包初始化
│   │   ├── generation.py                # LLM生成管理器
│   │   └── tensor_helper.py             # 张量辅助工具
│   ├── 📁 search/                        # 搜索模块
│   │   ├── build_index.sh               # 索引构建脚本
│   │   ├── google_search_server.py      # Google搜索服务器
│   │   ├── index_builder.py             # 索引构建器
│   │   ├── rerank_server.py             # 重排序服务器
│   │   ├── retrieval_request.py         # 检索请求定义
│   │   ├── retrieval_rerank_server.py   # 检索重排序服务器
│   │   ├── retrieval_server.py          # 检索服务器
│   │   ├── retrieval.py                 # 检索基础实现
│   │   ├── retrieval.sh                 # 检索启动脚本
│   │   └── serp_search_server.py        # SERP搜索服务器
│   └── __init__.py                      # 包初始化
│
├── 📁 verl/                              # 强化学习训练框架
│   ├── 📁 models/                        # 模型实现目录
│   │   ├── 📁 llama/                     # LLaMA模型实现
│   │   │   └── 📁 megatron/              # Megatron版本的LLaMA
│   │   │       ├── 📁 checkpoint_utils/ # 检查点工具
│   │   │       └── 📁 layers/           # 模型层实现
│   │   ├── 📁 transformers/              # Transformers兼容层
│   │   ├── __init__.py                  # 包初始化
│   │   ├── README_CN.md                 # 模型说明（中文）
│   │   ├── README.md                    # 模型说明（英文）
│   │   ├── registry.py                  # 模型注册器
│   │   └── weight_loader_registry.py    # 权重加载注册器
│   ├── 📁 single_controller/             # 单控制器模块
│   │   ├── 📁 base/                      # 基础控制器
│   │   │   ├── 📁 megatron/             # Megatron控制器
│   │   │   └── 📁 register_center/      # 注册中心
│   │   ├── 📁 ray/                       # Ray控制器
│   │   ├── 📁 version/                   # 版本控制
│   │   └── __init__.py                  # 包初始化
│   ├── 📁 third_party/                   # 第三方集成
│   │   └── 📁 vllm/                      # vLLM集成
│   │       ├── 📁 vllm_v_0_3_1/         # vLLM 0.3.1版本
│   │       ├── 📁 vllm_v_0_4_2/         # vLLM 0.4.2版本
│   │       ├── 📁 vllm_v_0_5_4/         # vLLM 0.5.4版本
│   │       └── 📁 vllm_v_0_6_3/         # vLLM 0.6.3版本
│   ├── 📁 trainer/                       # 训练器模块
│   │   ├── 📁 config/                    # 训练配置
│   │   │   ├── evaluation.yaml          # 评估配置
│   │   │   ├── generation.yaml          # 生成配置
│   │   │   ├── ppo_megatron_trainer.yaml # Megatron PPO配置
│   │   │   ├── ppo_trainer.yaml         # PPO训练配置
│   │   │   └── sft_trainer.yaml         # SFT训练配置
│   │   ├── 📁 ppo/                       # PPO算法实现
│   │   ├── __init__.py                  # 包初始化
│   │   ├── fsdp_sft_trainer.py          # FSDP SFT训练器
│   │   ├── main_eval.py                 # 主评估程序
│   │   ├── main_generation.py           # 主生成程序
│   │   ├── main_ppo_format.py           # PPO格式化训练程序
│   │   ├── main_ppo.py                  # PPO训练主程序
│   │   └── runtime_env.yaml             # 运行时环境配置
│   ├── 📁 utils/                         # 工具模块
│   │   ├── 📁 dataset/                   # 数据集工具
│   │   ├── 📁 debug/                     # 调试工具
│   │   ├── 📁 logger/                    # 日志工具
│   │   ├── 📁 megatron/                  # Megatron工具
│   │   ├── 📁 rendezvous/                # 分布式协调工具
│   │   ├── 📁 reward_score/              # 奖励评分工具
│   │   ├── __init__.py                  # 包初始化
│   │   ├── config.py                    # 配置工具
│   │   ├── distributed.py               # 分布式工具
│   │   ├── flops_counter.py             # FLOPS计数器
│   │   ├── fs.py                        # 文件系统工具
│   │   ├── fsdp_utils.py                # FSDP工具
│   │   ├── hdfs_io.py                   # HDFS IO工具
│   │   ├── import_utils.py              # 导入工具
│   │   ├── logging_utils.py             # 日志工具
│   │   ├── megatron_utils.py            # Megatron工具
│   │   ├── memory_buffer.py             # 内存缓冲区
│   │   ├── model.py                     # 模型工具
│   │   ├── py_functional.py             # Python函数式工具
│   │   ├── ray_utils.py                 # Ray工具
│   │   ├── seqlen_balancing.py          # 序列长度平衡工具
│   │   ├── tokenizer.py                 # 分词器工具
│   │   ├── torch_dtypes.py              # Torch数据类型工具
│   │   ├── torch_functional.py          # Torch函数式工具
│   │   ├── tracking.py                  # 跟踪工具
│   │   └── ulysses.py                   # Ulysses并行工具
│   ├── 📁 version/                       # 版本信息
│   ├── 📁 workers/                       # 工作节点模块
│   │   ├── 📁 actor/                     # Actor工作节点
│   │   ├── 📁 critic/                    # Critic工作节点
│   │   ├── 📁 reward_model/              # 奖励模型工作节点
│   │   │   └── 📁 megatron/             # Megatron奖励模型
│   │   ├── 📁 rollout/                   # Rollout工作节点
│   │   │   ├── 📁 naive/                # 简单Rollout
│   │   │   └── 📁 vllm_rollout/         # vLLM Rollout
│   │   ├── 📁 sharding_manager/          # 分片管理器
│   │   ├── __init__.py                  # 包初始化
│   │   ├── fsdp_workers.py              # FSDP工作节点
│   │   └── megatron_workers.py          # Megatron工作节点
│   ├── __init__.py                      # 包初始化
│   └── protocol.py                      # 协议定义
│
├── 📄 AGENTS.md                         # Agent说明文档
├── 📄 CLAUDE.md                         # Claude Code指导文档
├── 📄 infer.py                          # 推理主程序
├── 📄 LICENSE                           # 许可证
├── 📄 Notice.txt                        # 版权声明
├── 📄 pyproject.toml                    # 项目配置
├── 📄 README_CN.md                      # 项目说明（中文）
├── 📄 README.md                         # 项目说明（英文）
├── 📄 requirements.txt                  # 依赖要求
├── 📄 retrieval_launch.sh               # 检索服务启动脚本
├── 📄 setup.py                          # 安装脚本
├── 📄 train_grpo.sh                     # GRPO训练脚本
├── 📄 train_ppo.sh                      # PPO训练脚本
├── 📄 VERL_README_CN.md                 # veRL说明（中文）
└── 📄 VERL_README.md                    # veRL说明（英文）
```

## 核心架构

### 主要模块结构

- **search_r1/**: 核心搜索和推理模块
  - `llm_agent/generation.py`: LLM生成管理器，处理多轮推理和搜索交互
  - `search/retrieval.py`: 检索基础类和实现（BM25、密集检索）
  - `search/retrieval_server.py`: 基于FastAPI的检索服务器

- **verl/**: 强化学习训练框架
  - `trainer/main_ppo.py`: PPO训练主程序
  - `workers/`: Actor、Critic、Rollout等工作进程
  - `models/`: 模型实现（LLaMA、Qwen等）

- **scripts/**: 数据处理脚本
  - `data_process/nq_search.py`: NQ数据集处理
  - `download.py`: 数据和索引下载

### 核心类和接口

- **LLMGenerationManager**: 管理LLM生成过程和多轮推理-搜索循环
- **BaseRetriever**: 检索器抽象基类，支持BM25和密集检索
- **RayPPOTrainer**: 基于Ray的分布式PPO训练器

## 常用开发命令

### 环境安装

```bash
# 主环境
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb

# 检索环境（可选）
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### 数据准备

```bash
# 下载索引和语料
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# 处理NQ数据集
python scripts/data_process/nq_search.py
```

### 检索服务器

```bash
# 启动检索服务器
conda activate retriever
bash retrieval_launch.sh

# 或使用不同的检索器
bash example/retriever/retrieval_launch_bm25.sh  # BM25检索
bash example/retriever/retrieval_launch_google.sh  # Google搜索
```

### 训练命令

```bash
# PPO训练
conda activate searchr1
bash train_ppo.sh

# GRPO训练
bash train_grpo.sh

# 多节点训练
bash example/multinode/train_ppo_multinode_32b.sh
```

### 推理

```bash
# 启动检索服务器
conda activate retriever
bash retrieval_launch.sh

# 运行推理
conda activate searchr1
python infer.py
```

### 索引构建

```bash
# 构建自定义索引
bash search_r1/search/build_index.sh
```

## 训练配置

### 主要配置文件

- `verl/trainer/config/ppo_trainer.yaml`: PPO训练配置
- `verl/trainer/config/sft_trainer.yaml`: SFT训练配置
- `verl/trainer/config/evaluation.yaml`: 评估配置

### 关键参数

- `max_turns`: 最大交互轮数
- `data.max_prompt_length`: 最大提示长度
- `data.max_response_length`: 最大响应长度
- `data.max_obs_length`: 最大观察长度
- `retriever.url`: 检索服务器URL
- `retriever.topk`: 检索结果数量

## 数据格式

### QA数据格式

```python
data = {
    "data_source": data_source,
    "prompt": [{
        "role": "user",
        "content": question,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": solution
    },
    "extra_info": {
        'split': split,
        'index': idx,
    }
}
```

### 语料格式

```json
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 – July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
```

## 支持的搜索引擎

- 本地稀疏检索（BM25）
- 本地密集检索（FAISS Flat/ANN）
- 在线搜索引擎（Google、Bing、Brave等）
- 神经重排序器

## 支持的模型

- LLaMA系列（3B、8B、70B等）
- Qwen系列（3B、7B等）
- 支持FSDP和Megatron-LM分布式训练

## 开发注意事项

1. **环境管理**: 训练和检索使用不同的conda环境
2. **GPU内存**: 根据模型大小调整`gpu_memory_utilization`参数
3. **分布式训练**: 使用Ray进行分布式计算，支持多节点训练
4. **数据流**: 训练数据使用parquet格式，支持大规模数据处理
5. **日志**: 使用wandb进行实验跟踪

## 故障排除

- **vLLM + Qwen2**: 如果遇到问题，设置 `export VLLM_ATTENTION_BACKEND=XFORMERS`
- **内存不足**: 启用FSDP的参数卸载：`param_offload=true`
- **检索服务**: 确保检索服务器在8000端口运行
- **数据路径**: 检查数据文件路径是否正确配置