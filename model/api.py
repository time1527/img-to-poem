# add llm api here
# qwen

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
#     import os
#     import sys
#     sys.path.append("/home/dola")
#     from openai import OpenAI
#     from dotenv import find_dotenv,load_dotenv
#     _ = load_dotenv(find_dotenv())

#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2"),base_url = os.environ.get("OPENAI_BASE_URL2"))

#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content