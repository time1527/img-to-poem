import sys
sys.path.append("../")

import torch
from LOCALPATH import EMBEDDING_PATH,FT_EMBEDDING_PATH,RERANK_PATH
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def gen_ans(question,LLM):
    prompt = f"""你是一位诗人。根据文本做一句古诗，只返回古诗内容。
    
    文本：{question}

    有用的回答：
    """
    response = LLM(prompt)
    return response


def response_c(question,LLM,top_k = 5,query_poem=False):
    """
    question: image info + text
    query_poem: 微调的embedding的query是否是 白话文+文言文
    """
    if question is None or len(question) < 1:
            return ""
    try:
        if query_poem == True: # TODO
            # 0. default ans from llm
            default_ans = gen_ans(question,LLM)
            query = default_ans

            # 1.embedding and vector db 
            EMBEDDING = HuggingFaceEmbeddings(model_name=FT_EMBEDDING_PATH)
            FAISS_PERSIST_DIRECTORY = "./data/vectordb_select_c/faiss"
            vectordb = FAISS.load_local(FAISS_PERSIST_DIRECTORY, EMBEDDING)

            # 2.retrival
            contexts = vectordb.similarity_search(query,k=2*top_k)
            contexts = [c.page_content for c in contexts]

            # 3.reranker
            pairs = list(zip([query] * (2*top_k),contexts))

            tokenizer = AutoTokenizer.from_pretrained(RERANK_PATH)
            rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_PATH).to(DEVICE)
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                inputs = {key: inputs[key].cuda() for key in inputs.keys()}
                scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores_sorted,res_sorted = zip(*sorted(zip(scores, contexts)))
                
            res_sorted = res_sorted[::-1][:top_k]
        else:
            # 2024/3/1
            # 0. default ans from llm
            default_ans = gen_ans(question,LLM)

            # 1.embedding and vector db 
            EMBEDDING = HuggingFaceEmbeddings(model_name=FT_EMBEDDING_PATH)
            FAISS_PERSIST_DIRECTORY = "./data/vectordb_select_c/faiss"
            vectordb = FAISS.load_local(FAISS_PERSIST_DIRECTORY, EMBEDDING)

            # 2.retrival (doc, similarity)
            contexts = vectordb.similarity_search_with_score(question,k=top_k)

            # 3.
            res_sorted = [context[0].page_content for context in contexts]
            print(res_sorted)
        
        # 4.generate response(general)
        prompt = f"""请一步一步思考：
        首先检查上下文中是否存在与查询语义高度相符的项。
        如果存在，请只输出上下文中的最语义高度相符项，并在之后用括号注释：该内容来自古文检索。
        如果不存在，请根据查询输出一句古诗，并在之后用括号注释：该内容由AI生成。
        请严格按照输出要求进行输出，只需要输出古诗和括号注释内容，不需要输出思考过程。
        
        上下文:{res_sorted}

        查询:{question}

        默认项:{default_ans}

        有用的回答:
        """
        print(prompt)
        response = LLM(prompt)
        print(response)
        return response
    
    except Exception as e:
        return e
         

def response_ct(question,LLM,top_k = 5):
    """
    question: image info + text
    用于content-translation的检索

    将古文放在Document的metadata["content"]中，page_content为对应的翻译

    使用的RAG技巧：Hypothetical Questions、rerank
    question: 画面场景（现代白话文）
    context: 现代白话文
    
    question计算与page_content的相似度，再进行rerank————现代白话文与现代白话文之间的比较
    将rerank结果的metadata["content"]返回，即文言文部分
    
    LLM判断question和文言文之间的语义相似性，决定最终输出
    """
    if question is None or len(question) < 1:
            return ""
    try:
        # 1.embedding and vector db 
        EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
        FAISS_PERSIST_DIRECTORY = "./data/vectordb_ct/faiss"
        vectordb = FAISS.load_local(FAISS_PERSIST_DIRECTORY, EMBEDDING)
        
        # 2.retrival
        query = question
        contexts = vectordb.similarity_search(query,k=2*top_k)
        
        # 3.reranker translation-query
        pairs = list(zip([query] * (2*top_k),[c.page_content for c in contexts]))

        tokenizer = AutoTokenizer.from_pretrained(RERANK_PATH)
        rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_PATH).to(DEVICE)
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            inputs = {key: inputs[key].cuda() for key in inputs.keys()}
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy().tolist()
            scores_sorted,res_sorted = zip(*sorted(zip(scores, [c.metadata["content"] for c in contexts])))
            
        res_sorted = res_sorted[::-1][:top_k]
        
        # 4.generate response
        prompt = f"""请一步一步思考：
        首先检查上下文中是否存在与查询语义高度相符的项。
        如果存在，请只输出上下文中的最语义高度相符项，并在之后用括号注释：该内容来自古文检索。
        如果不存在，请根据查询输出一句古诗，并在之后用括号注释：该内容由AI生成。
        请严格按照输出要求进行输出，只需要输出古诗和括号注释内容，不需要输出思考过程。
        
        上下文:{res_sorted}

        查询:{query}

        有用的回答:
        """
        print(prompt)
        response = LLM(prompt)
        print(response)
        return response
        
    except Exception as e:
        return e