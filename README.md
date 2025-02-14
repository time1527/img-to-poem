# Image2Poem

## Question Description

**问题**：输入一张图返回一句诗/词/曲，$image \to text(poem)$。

**实现方式1**：VLM。

**实现方式2**：image2text检索，涉及到图像编码与诗词编码的对齐。

最朴素的方案：$image \to text[Chinese\ vernacular]\to text[poem]$检索。这可能出现两个gap：

* 图片-中文：生成文本可能不够丰富，针对这一点提供了文本框用于补充信息。

* 白话文-诗句，两种解决方案：
  * 向量数据库使用翻译-诗句对，类似Hypothetical Questions的形式存储，缺点是有明确对应翻译的诗句数量有限。可以使用现有「larger and stronger」LLM 来直接生成相应翻译；也可以使用已有数据对微调LLM，再来生成相应的翻译。
  
    >流程：输入图像 $\to$ 生成中文 $\to$ 检索翻译 $\to$ “取出”诗句
  
  * 使用所有诗句，微调嵌入模型：`{query:翻译,pos:[诗句文本],neg:[负样本(诗句))]}`，然后使用白话文作为query去检索诗句。
    
    >流程：输入图像 $\to$ 生成中文 $\to$​ 检索诗句 

将检索回的诗句、VLM生成的诗句作为候选对，重排序，以防“驴唇不对马嘴”，又能保证考虑到了现有的诗句。

## Run

**环境**：

```ps
conda create -n imgtopoem python=3.10 --y
conda activate imgtopoem
```

**option1**：VLM

```ps
pip install -r vlm_requirements.txt
python run_vlm.py
```

**option2**：all = Retrieval + VLM

```ps
pip install -r all_requirements.txt
```

修改settings.py；下载[数据集](https://github.com/VMIJUNV/chinese-poetry-and-prose)，并将“作品”解压到WORKS目录下。

```ps
python data.py
python run_all.py
```

## Embedding Finetune

> 2024/03/01微调，2025/02/13整理repo，归并到这里，原repo已本地备份。

**环境**：

```ps
conda create -n flagembedding python=3.10 --y
conda activate flagembedding
cd embedding-ft
git clone https://github.com/time1527/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
pip install faiss-cpu
pip install tensorboard
```

**数据准备**：

```ps
cd .. 
# 现在在embedding-ft目录下
```

准备hard negative mine 数据：

```ps
python data.py --function generate_candidate_pool
```

hard negative mine：以下命令使用了绝对路径

```ps
cd FlagEmbedding/scripts/
python hn_mine.py \
--input_file /home/pika/Project/img-to-poem/works/ct_works_qpn.jsonl \
--output_file /home/pika/Project/img-to-poem/works/minedHN.jsonl \
--candidate_pool /home/pika/Project/img-to-poem/works/candidate_pool.jsonl \
--range_for_sampling 2-20 \
--negative_number 9 \
--embedder_name_or_path /home/pika/Project/img-to-poem/models/bge-base-zh-v1.5
```

示例：

```json
{"query": "南山下田野里种植豆子，结果是草茂盛豆苗疏稀。", "pos": ["种豆南山下，草盛豆苗稀。"], "neg": ["种豆在南野，秫稻盈西畴。", "南亩种豆苗，苗稀草犹胜。", "南山尝种豆，碎荚落风雨。", "种豆在南山，种苗在东皋。", "东皋种禾禾渐焦，南山种豆枯豆苗。", "南山豆苗荒数亩，拂袖先归去，高官鼎内鱼，小吏罝中兔。", "闲来检点南山事，豆子苗生麦又齐。", "种蔬南冈下，地薄旱亦久。", "种田南山下，土薄良苗稀。"]}
```

划分训练、测试数据集：

```ps
cd ../..
# 现在在embeeding-ft目录下
python data.py --function split_data \
    --input_file /home/pika/Project/img-to-poem/works/minedHN.jsonl \
    --train_output_file /home/pika/Project/img-to-poem/works/train.jsonl \
    --test_output_file /home/pika/Project/img-to-poem/works/test.jsonl
```

**训练**：

```ps
torchrun --nproc_per_node 1 \
-m FlagEmbedding.finetune.embedder.encoder_only.base \
--model_name_or_path /home/pika/Model/bge-base-zh-v1.5 \
--cache_dir ./cache/model \
--output_dir /home/pika/Project/img-to-poem/models/ft-bge-zh-v1.5 \
--train_data /home/pika/Project/img-to-poem/works/train.jsonl \
--cache_path ./cache/data \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--normalize_embeddings True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 32 \
--train_group_size 10 \
--pad_to_multiple_of 8 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--warmup_ratio 0.1 \
--sentence_pooling_method cls \
--save_steps 1000 \
--logging_dir ./logs \
--report_to tensorboard
```

损失查看：

```ps
tensorboard --logdir ./logs
```

转换成SentenceTransformers模型：

```ps
python convert.py \
  --ckpt_dir /home/pika/Project/img-to-poem/models/ft-bge-zh-v1.5/checkpoint-3080 \
  --out_dir /home/pika/Project/img-to-poem/models/finetuned
```

**测试**：

准备数据集：

```ps
python data.py --function prepare_test_data \
    --candidate_file /home/pika/Project/img-to-poem/works/test.jsonl \
    --test_file /home/pika/Project/img-to-poem/works/test.jsonl
```

额外：

```ps
pip install pytrec_eval
```

测试微调前：

```ps
python test.py
```

测试微调后：

```ps
python test.py --model_path /home/pika/Project/img-to-poem/models/finetuned --task ft
```

| Metric   | Fine-tuned | Unfine-tuned |
| -------- | ---------- | ------------ |
| NDCG@1   | 0.91382    | 0.64159      |
| NDCG@5   | 0.95990    | 0.69277      |
| MAP@1    | 0.91382    | 0.64159      |
| MAP@5    | 0.94944    | 0.67962      |
| Recall@1 | 0.91382    | 0.64159      |
| Recall@5 | 0.98997    | 0.73142      |
| P@1      | 0.91382    | 0.64159      |
| P@5      | 0.19799    | 0.14628      |
| MRR@1    | 0.87460    | 0.61240      |
| MRR@5    | 0.92900    | 0.66480      |

## Example

Photo by [Marek Piwnicki](https://unsplash.com/@marekpiwnicki?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/a-snow-covered-mountain-range-with-a-clear-sky-Je7XqcBmDFg?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)      

![](./assets/example1_1.png)

![](./assets/example1_2.png)

## Timeline

2024-03-01：初版；embedding微调

2025-02-13：整理，添加VLM

## References

1. https://github.com/VMIJUNV/chinese-poetry-and-prose
2. https://github.com/FlagOpen/FlagEmbedding
3. https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6
