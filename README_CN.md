# Search-R1: 使用强化学习训练您的LLM进行推理和调用搜索引擎

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/论文1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/论文2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/资源-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/推特-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/日志-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>

**Search-R1** 是一个专为训练**推理与搜索交错的大型语言模型（LLM）**而设计的强化学习框架——这些语言模型能够学会以协调的方式进行推理和工具调用（例如调用搜索引擎）。

基于 [veRL](https://github.com/volcengine/verl) 构建，Search-R1 通过引入交错搜索引擎访问扩展了 **DeepSeek-R1(-Zero)** 的理念，并提供了完全开源的强化学习训练流程。它作为 **OpenAI DeepResearch** 的替代和开源解决方案，促进了工具增强的LLM推理的研究与开发。

我们支持不同的强化学习方法（如 PPO、GRPO、reinforce）、不同的LLM（如 llama3、Qwen2.5 等）以及不同的搜索引擎（如本地稀疏/密集检索器和在线搜索引擎）。

论文：[链接1](https://arxiv.org/pdf/2503.09516)、[链接2](https://arxiv.org/abs/2505.15117)；模型和数据：[链接](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5)；推特讨论：[链接](https://x.com/BowenJin13/status/1895544294473109889)；完整实验日志：[初步](https://wandb.ai/peterjin/Search-R1-open)、[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)、[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)。关于这些日志和方法的详细信息可以在[这里](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md)找到。

![single-turn](public/main.png)

## 新闻

- [2025.10] Search-R1 被 Thinking Machines Lab 的首个产品 [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) 收录！详情：[文档](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/tool_use/search)
- [2025.7] Search-R1 得到 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 的支持！详细说明：[代码](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/search)、[文档](https://novasky-ai.notion.site/skyrl-searchr1)
- [2025.6] Search-R1 现已集成到最新版本的 veRL 中，可以利用其最新功能！详细说明：[veRL](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)、[英文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md)、[中文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)
- [2025.5] 第二篇进行详细实证研究的[论文](https://arxiv.org/abs/2505.15117)发表，附带日志：[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)
- [2025.4] 我们支持 30B+ LLM 的[多节点](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/multinode.md)训练！
- [2025.4] 我们支持[不同的搜索引擎](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)，包括稀疏本地检索器、带 ANN 索引的密集本地检索器和在线搜索引擎！
- [2025.3] 第一篇 Search-R1 [论文](https://arxiv.org/pdf/2503.09516)发表，附带日志：[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)
- [2025.2] 我们开源 Search-R1 代码库，附带[初步结果](https://wandb.ai/peterjin/Search-R1-open)

## 链接

- [安装](#安装)
- [快速开始](#快速开始)
- [初步结果](#初步结果)
- [推理](#推理)
- [使用您自己的数据集](#使用您自己的数据集)
- [使用您自己的搜索引擎](#使用您自己的搜索引擎)
- [功能特性](#功能特性)
- [致谢](#致谢)
- [引用](#引用)

## 安装

### 两个conda环境备注
conda安装的包要注意默认的缓存位置。
让claude执行的时候，需要修改到数据盘。

### Search-r1 环境
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# 安装 torch [或者您可以跳过这一步，让 vllm 为您安装正确的版本]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 安装 vllm
pip3 install vllm==0.6.3 # 或者您可以安装 0.5.4、0.4.2 和 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation # 可以先不安装，在训练grpo的时候没有用这个
pip install wandb
```

### 检索器环境（可选）
如果您想调用本地检索器作为搜索引擎，可以按如下方式安装环境。（我们建议使用单独的环境。）
```bash
conda create -n retriever python=3.10
conda activate retriever

# 我们推荐使用 conda 安装 torch 以配合 faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## 安装 gpu 版本的 faiss 以保证高效的 RL 推理
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API 函数
pip install uvicorn fastapi
```

## 快速开始

在 NQ 数据集上使用 e5 作为检索器和 wikipedia 作为语料库训练推理 + 搜索 LLM。

(1) 下载索引和语料库。# 直接还用tavily代替
```bash
# 启动searchr1环境
# 这两份文件比较大，一个40G，一个20G
# 下载hf的大文件，都需要设置镜像网站以及access token
save_path=/root/autodl-tmp/data # 设置为数据盘
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 处理 NQ 数据集。
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate searchr1 && export HF_TOKEN=YOUR_HF_TOKEN && export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface_cache/datasets && export HF_ENDPOINT=https://hf-mirror.com && python scripts/data_process/nq_search.py  # 注意，nq_search.py中的路径已经修改/root/autodl-tmp/data/nq_search
#python scripts/data_process/nq_search.py
```

(3) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) 使用 qwen 运行 RL 训练（PPO）。
```bash
# 在RL脚本开头加上如下几行
# export HF_HOME=/root/autodl-tmp/huggingface_cache
# export HF_ENDPOINT=https://hf-mirror.com  # 国内加速
# export DATA_DIR='/root/autodl-tmp/data/nq_search'
conda activate searchr1
bash train_ppo.sh
```

## 初步结果

(1) 基础模型（llama3.2-3b-base）学会调用搜索引擎并获得性能提升。

![llama-3b](public/llama32-3b.png)


(2) 基础模型（Qwen2.5-7b-base）可以通过 RL 学会进行多轮搜索引擎调用和推理。

![multi-turn](public/multi-turn.png)

## 推理
#### 您可以使用训练好的 Search-R1 模型来回答您自己的问题。
(1) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) 运行推理。
```bash
conda activate searchr1
python infer.py
```
您可以修改第 7 行的 ```question``` 为您感兴趣的内容。

## 使用您自己的数据集

### 问答数据
对于每个问答样本，它应该是一个包含所需内容的字典，如下所示：

```
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

您可以参考 ```scripts/data_process/nq_search.py``` 获取具体的数据处理示例。

### 语料库

建议将您的语料库制作成一个 jsonl 文件，其中每一行（一个包含 "id" 键和 "contents" 键的字典）对应一个段落。您可以参考 ```example/corpus.jsonl``` 获取示例。

"id" 键对应段落 id，而 "contents" 键对应段落内容（'"' + title + '"\n' + text）。
例如：
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 – July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**为您的语料库建立索引（可选）。**
如果您想使用本地检索器作为搜索引擎，可以通过以下方式为您自己的语料库建立索引：
```
bash search_r1/search/build_index.sh
```
您可以更改 ```retriever_name``` 和 ```retriever_model``` 为您感兴趣的现成检索器。

## 使用您自己的搜索引擎

我们的代码库支持本地稀疏检索器（如 BM25）、本地密集检索器（支持 GPU 平面索引和 CPU ANN 索引）以及在线搜索引擎（如 Google、Bing 等）。更多细节可以在[这里](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md)找到。

主要理念是将本地或远程搜索引擎服务器与主要的 RL 训练流程分开启动。

LLM 可以通过调用搜索 API（例如 "http://127.0.0.1:8000/retrieve"）来调用搜索引擎。

您可以参考 ```search_r1/search/retriever_server.py``` 获取启动本地检索器服务器的示例。

## 功能特性
- 支持本地稀疏检索器（如 BM25）。 ✔️
- 支持本地密集检索器（平面索引和 ANN 索引） ✔️
- 支持 google search / bing search / brave search API 等。 ✔️
- 支持现成的神经重排序器。 ✔️
- 支持不同的 RL 方法（如 PPO、GRPO、reinforce）。 ✔️
- 支持不同的 LLM（如 llama3、Qwen2.5 等）。 ✔️

## 致谢

Search-R1 的概念受到 [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 和 [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main) 的启发。
其实现基于 [veRL](https://github.com/volcengine/verl) 和 [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main)。
我们衷心感谢这些团队为开源研究和开发所做的贡献。

## 由 Search-R1 支持或启发的优秀工作

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): 通过真实环境中的强化学习扩展深度研究。 [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1): 激励 LMM 进行搜索。 [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870): 通过强化学习实现最优工具调用。
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch): 在不搜索的情况下激励 LLM 的搜索能力。 [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1): 用于高效自适应搜索代理的强化内外知识协同推理。 [![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316): 通过信息觅食优化搜索增强推理。
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277): 思考过程中搜索和优化。 [![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582): 用于开放域开放性问题回答的基于搜索的代理模型。 [![[code]](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285): 增强代理搜索能力的通用预训练框架。 [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019): 基于视觉感知的 RAG 用于视觉丰富信息理解。 [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668): 通过 SFT 和 RL 训练 LLM 使用代码进行推理。 [![[code]](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185): 通过多奖励强化学习增强 LLM 推理。 [![[code]](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107): 通过逐步近端策略优化点燃 LLM 搜索能力。 [![[code]](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report): 多轮工具集成推理的稳定端到端强化学习。 [![[code]](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033): 通过强化学习教会 LLM 多轮路由和聚合。 [![[code]](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/): 用于 LLM 的模块化全栈 RL 库。 [![[code]](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976): 搜索代理的大规模 RL。 [![[code]](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303): 使用 RL 分解查询并并行搜索子查询。 [![[code]](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)
- [AutoTIR](https://arxiv.org/pdf/2507.21836): 通过强化学习实现自主工具集成推理。 [![[code]](https://img.shields.io/github/stars/weiyifan1023/AutoTIR)](https://github.com/weiyifan1023/AutoTIR)
- [verl-tool](https://arxiv.org/pdf/2509.01055): 支持多样化工具使用的 verl 版本。 [![[code]](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)](https://github.com/TIGER-AI-Lab/verl-tool)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240): LLM 代理强化学习的树搜索。 [![[code]](https://img.shields.io/github/stars/AMAP-ML/Tree-GRPO)](https://github.com/AMAP-ML/Tree-GRPO)




## 引用

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```