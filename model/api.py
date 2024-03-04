# add llm api here
# qwen
import os
import sys
from LOCALPATH import ENV_PATH
sys.path.append(ENV_PATH)
from dotenv import find_dotenv,load_dotenv
_ = load_dotenv(find_dotenv())

def qwen(prompt,model="qwen-max"):
    from http import HTTPStatus
    import dashscope

    response = dashscope.Generation.call(
        model=model,
        prompt=prompt
    )
    if response.status_code == HTTPStatus.OK:
        return response.output["text"]
    else:
        return response.message
    
# openai


# def openai(prompt,model = "gpt-3.5-turbo-1106"):
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2"),base_url = os.environ.get("OPENAI_BASE_URL2"))

#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content