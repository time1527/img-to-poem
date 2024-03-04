import gradio as gr

import sys
sys.path.append("../")
from model.api import qwen
from tools.imgtotext import generate_from_img
from tools.response import response_ct,response_c

LLM = qwen

def generate(img,text,data):
    if img != None:
        img_info = generate_from_img(img,LLM)
    else:
        img_info = None

    if img_info and text:
        question = "。".join([img_info,text]) 
    elif img_info:
        question = img_info
    elif text:
        question = text
    else:
        return "请输入图片或者文本"

    if data == "content-translation":
        response = response_ct(question,LLM)
    else:
        response = response_c(question,LLM)
    return response


with gr.Blocks() as demo:
    with gr.Row(equal_height = True):   
        gr.Markdown("""<h1><center>ImgtoPoem</center></h1>""")
    with gr.Row():
        source_img = gr.Image(type="filepath",label = "Upload Image")
    with gr.Row():
        text = gr.Textbox(label = "Text",info="additional details",interactive = True)
        select_data = gr.Dropdown(["content-translation", "content"], label="Data", info="database to retrieve")
    with gr.Row():
        gen_btn = gr.Button("Generate Poem")
    with gr.Row():
        output_text = gr.Textbox(label = "Output",scale=2)
    
    gen_btn.click(fn = generate,
                  inputs=[source_img,text,select_data],
                  outputs=output_text)
gr.close_all()
demo.launch()