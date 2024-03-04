# ImgtoPoem

## 写在前面

理想中的实现是输入一张图之后会直接返回一句诗/词/曲（检索优先，如果检索不到再AI生成），即图片 $\Rightarrow$ 诗句。

但在实践中，暂时的实现是图片 $\Rightarrow$ 英文 $\Rightarrow$ 中文（现代白话文） $\Rightarrow$ 诗句。这中间可能出现两个gap：

* 图片-中文，生成文本可能不够丰富，针对这一点提供了文本框用于补充信息
* 白话文-诗句，暂时两种解决方案：
  * 使用有翻译的诗句，对应文件 `ct_data.jsonl`，类似Hypothetical Questions的形式存储数据库，缺点是根据数据格式有明确对应翻译的诗句数量有限
  * 使用所有诗句，对应文件 `c_data.jsonl`，具体而言又有两种方案
    * 微调嵌入模型：`{query:翻译,pos:[诗句文本],neg:[负样本(诗句))]}`，然后使用白话文作为query去索引诗句
    * 微调嵌入模型：`{query:翻译+llm生成的诗句,pos:[诗句文本],neg:[负样本(诗句))]}`，在查询时，先让大模型生成一个答案，然后查询+答案拼接去向量数据库检索诗句，或者 `{query:llm生成的诗句,pos:[诗句文本],neg:[负样本(诗句))]}`

目前重点放在了“白话文-诗句”gap的解决，但由于“image to poem”是最初的构想，上传图片部分没有取消。

## 运行

1. 在 `LOCALPATH.py`内指定本地路径，主要是RERANK_PATH和EMBEDDING_PATH
2. 如果不想使用图生文模型，可直接在文本框内填写相关信息
3. `python run.py`

## 结构

**说明：[]表示不在repo中但可由“来源”或者python文件（在此目录中）得到的内容**

### data

```bash
data
├── [works.json]: 数据来自https://github.com/VMIJUNV/chinese-poetry-and-prose
├── [c_data.jsonl]: 由prepare_data.ipynb和works.json生成
├── [ct_data.jsonl]: 由prepare_data.ipynb和works.json生成
├── prepare_c_db.ipynb
├── prepare_ct_db.ipynb
├── prepare_data.ipynb
├── [select_c_data.jsonl]: 由prepare_data.ipynb和works.json生成，c_data.jsonl的替代
├── vectordb_ct
│   └── faiss
│       ├── index.faiss
│       └── index.pkl
└── vectordb_select_c
    └── faiss
        ├── index.faiss
        └── index.pkl
```

### tools

```bash
tools
├── imgtotext.py: 图转文
└── response.py
```

## 结果
原图片来自: https://unsplash.com/t/nature

1. https://unsplash.com/photos/a-snow-covered-mountain-range-with-a-clear-sky-Je7XqcBmDFg?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash

   Photo by <a href="https://unsplash.com/@eugene_golovesov?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Eugene Golovesov</a> on <a href="https://unsplash.com/photos/a-group-of-trees-that-are-in-the-snow-z994gPo74ck?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  
  ![屏幕截图 2024-03-04 194838](https://github.com/time1527/img-to-poem/assets/154412155/5370ad61-c04a-4760-9ed0-849497e3f12d)

  ![屏幕截图 2024-03-04 194816](https://github.com/time1527/img-to-poem/assets/154412155/e29f2832-d697-4a5a-afad-37216e2d6c49)

2. https://unsplash.com/photos/a-snow-covered-mountain-range-with-a-clear-sky-Je7XqcBmDFg?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash

   Photo by <a href="https://unsplash.com/@marekpiwnicki?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Marek Piwnicki</a> on <a href="https://unsplash.com/photos/a-snow-covered-mountain-range-with-a-clear-sky-Je7XqcBmDFg?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

   ![屏幕截图 2024-03-04 194910](https://github.com/time1527/img-to-poem/assets/154412155/269f425c-3b74-4c89-b231-ae9aec53412b)

   ![屏幕截图 2024-03-04 200055](https://github.com/time1527/img-to-poem/assets/154412155/b7c8f10a-59b2-4078-869a-e28d62c6ad50)

## 其他
[微调Embedding](https://github.com/time1527/finetune/tree/main/translation-bge-zh-v1.5)

[微调LLM](https://github.com/time1527/finetune/tree/main/translation-chatglm3-6b)

## TODO

* [ ] reranker微调
* [ ] 尝试[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

## 参考
1. https://github.com/VMIJUNV/chinese-poetry-and-prose
2. https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6
3. https://unsplash.com/t/nature
