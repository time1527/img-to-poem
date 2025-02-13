import os
import re
import sys
import json
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from FlagEmbedding import FlagModel
from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
import argparse

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from settings import WORKS, EMBEDDING, FT_EMBEDDING


def test(model_path, task, queries_file, corpus_file, qrels_file):

    model = FlagModel(
        model_path,
        query_instruction_for_retrieval="",
        devices=[0],
        use_fp16=False,
    )

    queries = load_dataset("json", data_files=queries_file)["train"]
    corpus = load_dataset("json", data_files=corpus_file)["train"]
    qrels = load_dataset("json", data_files=qrels_file)["train"]

    qids_in_qrels = set([line["qid"] for line in qrels])
    queries_text = [query["text"] for query in queries if query["id"] in qids_in_qrels]
    corpus_text = [sub["text"] for sub in corpus]

    qrels_dict = {}
    for line in qrels:
        if line["qid"] not in qrels_dict:
            qrels_dict[line["qid"]] = {}
        qrels_dict[line["qid"]][line["docid"]] = line["relevance"]

    print(
        f"queries_text: {len(queries_text)}, corpus_text: {len(corpus_text)}, qrels: {len(qrels_dict.items())}"
    )

    queries_embeddings = model.encode_queries(queries_text)
    corpus_embeddings = model.encode_corpus(corpus_text)

    # create and store the embeddings in a Faiss index
    dim = corpus_embeddings.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)

    query_size = len(queries_embeddings)

    all_scores = []
    all_indices = []

    # search top 10 answers for all the queries
    for i in tqdm(range(0, query_size, 32), desc="Searching"):
        j = min(i + 32, query_size)
        query_embedding = queries_embeddings[i:j]
        score, indice = index.search(query_embedding.astype(np.float32), k=10)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # store the results into the format for evaluation
    results = {}
    for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
        results[queries["id"][idx]] = {}
        for score, index in zip(scores, indices):
            if index != -1:
                try:
                    results[queries["id"][idx]][corpus["id"][index]] = float(score)
                except:
                    print(f"index: {index}, score: {score}")

    k_values = [1, 5]
    eval_res = evaluate_metrics(qrels_dict, results, k_values)
    mrr = evaluate_mrr(qrels_dict, results, k_values)
    with open(f"./logs/{task}-test_results.txt", "w") as f:
        f.write(f"eval_res: {eval_res}\n")
        f.write(f"mrr: {mrr}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=EMBEDDING,
        help="Path to the model to be evaluated.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="unft",
        help="Task name for the evaluation.",
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=os.path.join(WORKS, "test_queries.jsonl"),
        help="Path to the queries file.",
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=os.path.join(WORKS, "corpus.jsonl"),
        help="Path to the corpus file.",
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        default=os.path.join(WORKS, "test_qrels.jsonl"),
        help="Path to the qrels file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(
        args.model_path, args.task, args.queries_file, args.corpus_file, args.qrels_file
    )
