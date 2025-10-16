# 数据集格式
## RLHF 数据集
我们将所有数据源组合成一个单一的 parquet 文件。我们直接将提示组织成聊天格式，以便轻松融入多轮聊天。在提示中，我们可能会添加指令遵循文本来引导模型以特定格式输出答案，以便我们能够提取答案。

数学问题
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia 在 4 月份向她的 48 个朋友卖了夹子，然后在 5 月份卖了一半数量的夹子。Natalia 在 4 月和 5 月总共卖了多少个夹子？让我们一步一步思考，并在 \"####\" 后输出最终答案"}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```