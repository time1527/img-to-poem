import os

current_file_path = os.path.abspath(__file__)
current_directory_path = os.path.dirname(current_file_path)

WORKS = os.path.join(current_directory_path, "works")
VB = os.path.join(current_directory_path, "vb")

EMBEDDING = os.path.join(current_directory_path, "models/bge-base-zh-v1.5")
FT_EMBEDDING = os.path.join(current_directory_path, "models/finetuned")
RERANKER = os.path.join(current_directory_path, "models/bge-reranker-base")
