import os
import sys
from settings import EMBEDDING, FT_EMBEDDING, RERANKER, WORKS, VB
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)
from vlm import generate_poem, image_caption
from db import get_ct_db, get_select_c_db


def retrieve_content(image, text=None, top_k=5):
    if image is None:
        return "请输入图像。"
    try:
        text = "" if text is None else text

        # 2024/3/1
        # 0. default ans from vlm
        default_ans = generate_poem(image)
        print(default_ans)

        # 1.embedding and vector db
        embedding = HuggingFaceEmbeddings(
            model_name=FT_EMBEDDING,
            encode_kwargs={"normalize_embeddings": True},
        )
        faiss_path = os.path.join(VB, "select_c_db")
        if not os.path.exists(faiss_path):
            get_select_c_db(FT_EMBEDDING)
        vectordb = FAISS.load_local(
            faiss_path, embedding, allow_dangerous_deserialization=True
        )

        # 2.retrieval (doc, similarity)
        if image is not None:
            query = image_caption(image) + text
        else:
            query = text
        print(query)
        contexts = vectordb.similarity_search_with_score(query, k=top_k)

        # 3.sorted
        candidates = list(
            set([context[0].page_content for context in contexts] + [default_ans])
        )

        # 4.rerank:default ans or retrieval ans
        reranker = HuggingFaceCrossEncoder(model_name=RERANKER)
        scores = reranker.score([(query, candicate) for candicate in candidates])
        sorted_candidates = [
            c for _, c in sorted(zip(scores, candidates), reverse=True)
        ]
        ans = sorted_candidates[:2]
        ans = [
            f"{c}（使用检索）" if c != default_ans else f"{c}（使用VLM）" for c in ans
        ]
        return "\n".join(ans)
    except Exception as e:
        return e


def retrieve_translation(image, text=None, top_k=5):
    """
    content-translation的检索
    将古文放在Document的metadata["content"]中，page_content为对应的翻译
    使用的RAG技巧：Hypothetical Questions
    """
    if image is None:
        return "请输入图像。"
    try:
        text = "" if text is None else text
        # 0.default ans from vlm
        default_ans = generate_poem(image)
        print(default_ans)

        # 1.embedding and vector db
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING,
            encode_kwargs={"normalize_embeddings": True},
        )
        faiss_path = os.path.join(VB, "ct_db")
        if not os.path.exists(faiss_path):
            get_ct_db(EMBEDDING)
        vectordb = FAISS.load_local(
            faiss_path, embedding, allow_dangerous_deserialization=True
        )

        # 2.retrieval
        if image is not None:
            query = image_caption(image) + text
        else:
            query = text
        print(query)
        contexts = vectordb.similarity_search(query, k=top_k)

        # 3.reranker translation-query
        reranker = HuggingFaceCrossEncoder(model_name=RERANKER)
        candidates = list(
            set([context.metadata["content"] for context in contexts] + [default_ans])
        )

        scores = reranker.score([(query, candicate) for candicate in candidates])
        # 4.sorted
        sorted_candidates = [
            c for _, c in sorted(zip(scores, candidates), reverse=True)
        ]
        ans = sorted_candidates[:2]
        ans = [
            f"{c}（使用检索）" if c != default_ans else f"{c}（使用VLM）" for c in ans
        ]
        return "\n".join(ans)

    except Exception as e:
        return e
