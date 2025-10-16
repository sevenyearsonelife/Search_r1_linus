# Repository Guidelines

## 项目结构与模块组织
Search-R1 主干代码位于 `search_r1/`：`llm_agent/` 聚焦推理与工具调用策略，`search/` 提供检索服务器、索引构建与重排脚本。veRL 基础设施保存在 `verl/`，其中 `trainer/`、`workers/`、`utils/` 切分清晰，扩展算法时请限定在对应子模块并同步更新 `verl/version/`。`scripts/` 下集中数据预处理与实验驱动脚本，`docs/`、`public/`、`example/` 分别存放文档、可视化素材与部署模板，遵循原有布局可避免交叉影响。顶层 `README.md` 与 `VERL_README.md` 提供环境搭建差异说明，新增指南时请链接到对应章节以保持导航一致。依赖声明集中在 `requirements.txt` 与 `pyproject.toml`，如需新增库请先评估 GPU/CUDA 兼容性并更新两处记录。

```text
Search-R1/
├── search_r1/                 搜索与智能体核心库
│   ├── llm_agent/             推理策略、生成辅助与张量工具
│   └── search/                检索服务端、索引构建与重排脚本
├── verl/                      veRL 强化学习训练框架
│   ├── models/                模型注册与权重加载实现
│   ├── trainer/               训练入口、配置与 PPO/GRPO 主循环
│   ├── workers/               actor/critic/rollout 等分布式工作单元
│   ├── utils/                 分布式、日志、数据集等通用工具库
│   ├── single_controller/     单控制器调度组件
│   ├── third_party/           外部依赖适配层
│   └── version/               版本记录与发布元数据
├── scripts/                   数据预处理与批量任务脚本
│   ├── data_process/          通用数据清洗与格式转换流水线
│   ├── nq_hotpotqa/           NQ/HotpotQA 专用处理脚本
│   ├── download.py|.sh        下载模型或数据资产
│   └── upload.py|.sh          上传检查点或日志到远端
├── docs/                      项目使用、部署与实验记录文档
├── example/                   示例语料、配置与多机部署模板
├── public/                    展示所需静态资源（图片等）
├── infer.py                   推理入口脚本
├── retrieval_launch.sh        检索服务一键启动脚本
├── train_ppo.sh               PPO 训练脚本（单机多卡默认参数）
├── train_grpo.sh              GRPO 训练脚本
├── README.md | README_CN.md   顶层说明文档（中英版）
├── VERL_README.md|_CN.md      veRL 独立说明文档
├── requirements.txt           Python 依赖清单
└── pyproject.toml | setup.py  构建与打包配置
```

## 构建、测试与开发命令
- `python -m venv .venv && source .venv/bin/activate`：创建隔离虚拟环境，避免全局包冲突，激活后再执行后续步骤。
- `pip install -e .[test]`：在根目录安装本地包及测试依赖，确保 veRL 扩展可被 Python 解析并具备 `pytest`/`yapf`。
- `python infer.py --config docs/example_infer.yaml`：使用示例配置快速验证推理链路，可替换为自定义 YAML 并通过 `--save_dir outputs/demo` 指定结果目录。
- `bash train_ppo.sh` / `bash train_grpo.sh`：默认读取脚本内部环境变量，适合单机 8 卡实验，修改前请复制脚本避免污染主线，并保留 `tee` 生成的日志以便回溯。
- `bash retrieval_launch.sh --mode local`：统一启动检索服务；也可参考 `example/retriever/retrieval_launch_bm25.sh` 或 `retrieval_launch_google.sh` 以切换搜索后端，并确保 `SERPAPI_KEY` 已写入环境变量。

## 编码风格与命名约定
Python 模块保持 4 空格缩进、`snake_case` 文件与函数命名、类名使用 `CamelCase`。提交前使用 `yapf` 或 IDE 内置格式化器对齐导入顺序与行宽（建议 120 列），核心模块需补充类型注解与 docstring 说明输入输出维度。配置文件以 YAML/JSON 呈现，键名采用小写连字符或下划线并与目录名称一致；新增脚本需说明入口参数并保持 `if __name__ == "__main__":` 守卫，命令行参数建议通过 `argparse` 声明以便自动生成帮助文本。

## 测试指南
仓库默认使用 `pytest`，请在根目录运行 `pytest -q` 以获得最简输出；CI 环境中建议改为 `pytest --maxfail=1 --durations=10` 以快速捕获槽点。缺失的测试目录可在 `tests/` 下按模块镜像结构创建，例如 `tests/search/test_retrieval.py`，并使用 `fixtures` 复用假数据。新增特性需覆盖主流程与极端输入，若依赖外部搜索服务，请使用 `pytest.mark.integration` 标记并在文档中注明运行前提。目标语句覆盖率保持在 70% 以上；涉及 RL 训练的长链路，可通过 Mock retriever 或截断步骤缩短执行时间。提交前确认关键路径（检索、策略生成、训练循环）通过至少一次本地测试，并把测试指令写入 PR 描述。

## 提交与 Pull Request 要求
Git 历史偏好简洁祈使句风格，如 `add skyrl news`、`fix typos`；请保持 60 字符以内并突出动作对象。推荐使用 `feature/<topic>` 或 `bugfix/<issue-id>` 作为分支命名，便于追踪。Pull Request 描述需包含变更摘要、影响模块、测试结果以及相关 Issue/实验链接，涉及脚本或配置变动时附带示例命令与预期资源消耗。若修改跨越 `search_r1/` 与 `verl/`，请在清单中按子模块列出影响面。确保合并前已完成自检并标明潜在风险（如额外 GPU 需求或外部 API 额度），必要时 @ 相关模块维护者。

## 安全与配置提示
搜索与推理依赖的 API Key、检索索引路径务必通过环境变量或未纳入版本控制的配置文件传递，勿提交到仓库。运行 `retrieval_server.py` 等长期进程时，请复用 `screen`/`tmux` 并限制日志输出到 `logs/` 子目录，避免覆盖核心训练日志；日志轮转可通过 `--log_dir` 参数或 `logrotate` 脚本管理。敏感配置若需分享，请在文档中提供脱敏样例而非真实凭据，并检查 `.gitignore` 已包含临时索引与缓存目录。推荐借助 `python-dotenv` 等工具在本地加载凭据，同时部署到云端前核对出口流量策略，防止无鉴权的检索 API 对外暴露。
