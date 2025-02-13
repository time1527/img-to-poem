import os
import re
import sys
import json
import random
import argparse

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from settings import WORKS


def generate_candidate_pool():
    ct_path = os.path.join(WORKS, "ct_works.jsonl")
    c_path = os.path.join(WORKS, "c_works.jsonl")
    candidate_path = os.path.join(WORKS, "candidate_pool.jsonl")
    ct_qpn_path = os.path.join(WORKS, "ct_works_qpn.jsonl")

    if os.path.exists(candidate_path):
        os.remove(candidate_path)
    if os.path.exists(ct_qpn_path):
        os.remove(ct_qpn_path)

    with open(ct_path, "r") as f_in, open(ct_qpn_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)  # 将 JSON 字符串转换为 Python 对象
            query = data["translation"]
            pos = data["content"]
            json.dump(
                {"query": query, "pos": [pos], "neg": []}, f_out, ensure_ascii=False
            )
            f_out.write("\n")
    print("ct_qpn done")

    with open(c_path, "r") as f_in, open(
        candidate_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            data = json.loads(line)  # 将 JSON 字符串转换为 Python 对象
            text = data["content"]
            json.dump({"text": text}, f_out, ensure_ascii=False)
            f_out.write("\n")
    print("candidate done")


def split_data(input_file, train_output_file, test_output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        # Read the file line by line and parse each line as a JSON object
        data = [json.loads(line) for line in file]

    num = len(data)

    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

    split_point = int(num * 0.9)

    train_data = data[:split_point]
    test_data = data[split_point:]

    print(f"train data size = {len(train_data)}, train data example = {train_data[0]}")
    print(f"test data size = {len(test_data)}, test data example = {test_data[0]}")

    with open(train_output_file, "w", encoding="utf-8") as f:
        for d in train_data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    with open(test_output_file, "w", encoding="utf-8") as f:
        for d in test_data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
    print(
        f"Split complete. Train data written to {train_output_file}, Test data written to {test_output_file}"
    )


def prepare_test_data(candidate_file, test_file):

    # 输出文件路径
    corpus_file = os.path.join(WORKS, "corpus.jsonl")
    test_queries_file = os.path.join(WORKS, "test_queries.jsonl")
    test_qrels_file = os.path.join(WORKS, "test_qrels.jsonl")

    if os.path.exists(corpus_file):
        os.remove(corpus_file)
    if os.path.exists(test_queries_file):
        os.remove(test_queries_file)
    if os.path.exists(test_qrels_file):
        os.remove(test_qrels_file)

    # 读取候选池数据并创建 corpus
    corpus = {}
    corpus_id = 1  # 从 1 开始编号
    with open(candidate_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus[str(corpus_id)] = data["pos"][0]
            corpus_id += 1
            for neg_text in data.get("neg", []):
                corpus[str(corpus_id)] = neg_text
                corpus_id += 1

    # 写入 corpus.jsonl
    with open(corpus_file, "w", encoding="utf-8") as f:
        for corpus_id, text in corpus.items():
            json.dump(
                {"id": corpus_id, "title": "", "text": text}, f, ensure_ascii=False
            )
            f.write("\n")

    # 读取测试数据并创建 test_queries 和 test_qrels
    test_queries = []
    test_qrels = []

    with open(test_file, "r", encoding="utf-8") as f:
        for qid, line in enumerate(f):
            data = json.loads(line)
            query_id = str(qid + 1)  # 从 1 开始编号
            test_queries.append({"id": query_id, "text": data["query"]})

            for pos_text in data.get("pos", []):
                clean_pos_text = pos_text
                for doc_id, doc_text in corpus.items():
                    if doc_text == clean_pos_text:
                        test_qrels.append(
                            {"qid": query_id, "docid": doc_id, "relevance": 1}
                        )
                        break

    # 写入 test_queries.jsonl
    with open(test_queries_file, "w", encoding="utf-8") as f:
        for query in test_queries:
            json.dump(query, f, ensure_ascii=False)
            f.write("\n")
    print("test_queries done---{}".format(len(test_queries)))
    # 写入 test_qrels.jsonl
    with open(test_qrels_file, "w", encoding="utf-8") as f:
        for qrel in test_qrels:
            json.dump(qrel, f, ensure_ascii=False)
            f.write("\n")
    print("test_qrels done---{}".format(len(test_qrels)))


def main():
    parser = argparse.ArgumentParser(
        description="Run specific functions based on command line arguments."
    )
    parser.add_argument(
        "--function",
        choices=["generate_candidate_pool", "split_data", "prepare_test_data"],
        required=True,
        help="Specify the function to run.",
    )
    parser.add_argument(
        "--input_file", required=False, help="Input file for split_data function."
    )
    parser.add_argument(
        "--train_output_file", required=False, help="Output file for training data."
    )
    parser.add_argument(
        "--test_output_file", required=False, help="Output file for testing data."
    )
    parser.add_argument(
        "--candidate_file",
        required=False,
        help="Candidate file for prepare_test_data function.",
    )
    parser.add_argument(
        "--test_file", required=False, help="Test file for prepare_test_data function."
    )

    args = parser.parse_args()

    if args.function == "generate_candidate_pool":
        generate_candidate_pool()
    elif args.function == "split_data":
        split_data(args.input_file, args.train_output_file, args.test_output_file)
    elif args.function == "prepare_test_data":
        prepare_test_data(args.candidate_file, args.test_file)


if __name__ == "__main__":
    main()
