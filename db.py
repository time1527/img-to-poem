import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)
from settings import EMBEDDING, WORKS, VB, FT_EMBEDDING


def get_select_c_db(embedding_path):
    # Define the metadata extraction function.
    def metadata_func(record: dict, metadata: dict) -> dict:

        # metadata['author'] = record.get('author')
        # metadata['title'] = record.get('title')

        # return metadata
        return {}

    loader = JSONLoader(
        file_path=os.path.join(WORKS, "select_c_works.jsonl"),
        jq_schema=".content",
        metadata_func=metadata_func,
        text_content=False,
        json_lines=True,
    )

    docs = loader.load()
    print(docs[1000])

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_path,
        encode_kwargs={"normalize_embeddings": True},
    )

    faiss_db = FAISS.from_documents(
        documents=docs,
        embedding=embedding,
    )
    faiss_db.save_local(os.path.join(VB, "select_c_db"))
    print("save db done")


def get_ct_db(embedding_path):
    # Define the metadata extraction function.
    def metadata_func(record: dict, metadata: dict) -> dict:
        # return {}
        metadata["content"] = record.get("content")
        return metadata

    loader = JSONLoader(
        file_path=os.path.join(WORKS, "ct_works.jsonl"),
        jq_schema=".",
        content_key="translation",  # Hypothetical Questions
        metadata_func=metadata_func,
        text_content=False,
        json_lines=True,
    )
    docs = loader.load()
    print(docs[1000])

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_path,
        encode_kwargs={"normalize_embeddings": True},
    )

    faiss_db = FAISS.from_documents(documents=docs, embedding=embedding)

    faiss_db.save_local(os.path.join(VB, "ct_db"))
    print("save db done")
