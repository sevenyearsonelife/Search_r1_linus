## 搜索引擎

在本文档中，我们提供了如何启动不同检索器的示例，包括本地稀疏检索器（如 BM25）、本地密集检索器（如 e5）和在线搜索引擎。
对于本地检索器，我们使用 [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) 语料库作为示例，语料库索引可以在以下位置找到：[bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index)、[e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)、[e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64)。

### 如何选择检索器？

- 如果您有私有或特定领域的语料库，请选择 **本地检索器**。

    - 如果您的领域没有高质量的基于嵌入的检索器（密集检索器），请选择 **稀疏本地检索器**（如 BM25）。

    - 否则选择 **密集本地检索器**。

        - 如果您没有足够的 GPU 来进行精确的密集嵌入匹配，请选择在 CPU 上的 **ANN 索引**。

        - 如果您有足够的 GPU，请选择在 GPU 上的 **平面索引**。


- 如果您想训练一个通用的 LLM 搜索代理并且有足够的资金，请选择 **在线搜索引擎**（如 [SerpAPI](https://serpapi.com/)）。


- 如果您有特定领域的在线搜索引擎（如 PubMed 搜索），您可以参考[链接](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py)自行将其集成到 Search-R1 中。

搜索引擎启动脚本可以在[链接](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever)中找到。

### 本地稀疏检索器

稀疏检索器（如 bm25）是一种传统方法。检索过程非常高效，不需要 GPU。但是，在某些特定领域，它可能不如密集检索器准确。

(1) 下载索引。
```bash
save_path=/您的/保存/路径
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

(2) 启动本地 BM25 检索器服务器。
```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```


### 本地密集检索器

您也可以采用一些现成的密集检索器，例如 e5。在某些特定领域，这些模型比稀疏检索器强大得多。
如果您有足够的 GPU，我们推荐使用下面的平面索引变体，否则您可以采用 ANN 变体。

#### 平面索引

平面索引进行精确的嵌入匹配，速度慢但非常准确。为了使其足够高效以支持在线强化学习，我们建议通过 ```--faiss_gpu``` 启用 **GPU** 使用。

(1) 下载索引和语料库。
```bash
save_path=/保存/路径
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 启动本地平面 e5 检索器服务器。

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu

```


#### ANN 索引（HNSW64）

为了仅使用 **CPU** 提高搜索效率，您可以采用近似最近邻（ANN）索引，例如使用 HNSW64。
它非常高效，但可能不如平面索引准确，特别是当检索到的段落数量较小时。

(1) 下载索引。
```bash
save_path=/保存/路径
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```


(2) 启动本地 ANN 密集检索器服务器。
```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```


### 在线搜索引擎

我们同时支持 [Google Search API](https://developers.google.com/custom-search/v1/overview) 和 [SerpAPI](https://serpapi.com/)。我们推荐 [SerpAPI](https://serpapi.com/)，因为它集成了多个在线搜索引擎 API（包括 Google、Bing、百度等），并且没有月度配额限制（[Google Search API](https://developers.google.com/custom-search/v1/overview) 有严格的 10k 月度配额，不足以满足在线 LLM 强化学习训练的需求）。

#### SerAPI 在线搜索服务器

```bash
search_url=https://serpapi.com/search
serp_api_key="" # 在这里放入您的 serp api 密钥（https://serpapi.com/）

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google 在线搜索服务器

```bash
api_key="" # 在这里放入您的 google 自定义 API 密钥（https://developers.google.com/custom-search/v1/overview）
cse_id="" # 在这里放入您的 google cse API 密钥（https://developers.google.com/custom-search/v1/overview）

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```