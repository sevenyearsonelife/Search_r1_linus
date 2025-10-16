## 复现论文结果

### 下载数据集

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/nq_hotpotqa_train
```

### 启动本地搜索引擎

(1) 下载索引和语料库。
```bash
save_path=/保存/路径
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

### 运行 PPO 训练
```bash
bash train_ppo.sh
```


### 运行 GRPO 训练
```bash
bash train_grpo.sh
```

### 运行评估
```bash
bash evaluate.sh
```

您可以将 ```$BASE_MODEL``` 更改为您想要评估的模型路径。