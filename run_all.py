import os
import sys
import gradio as gr

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)
from retrieval import rag_content, rag_translation


def generate(image, text, select_data):
    if select_data == "content":
        return rag_content(image, text)
    else:
        return rag_translation(image, text)


with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        gr.Markdown("""<h1><center>ImagetoPoem</center></h1>""")
    with gr.Row():
        source_img = gr.Image(
            type="filepath", label="Upload Image", height=600, width=600
        )
    with gr.Row():
        text = gr.Textbox(label="Text", info="additional details", interactive=True)
        select_data = gr.Dropdown(
            ["content-translation", "content"],
            label="Data",
            info="database to retrieve",
        )
    with gr.Row():
        gen_btn = gr.Button("Generate Poem")
    with gr.Row():
        output_text = gr.Textbox(label="Output", scale=2)

    gen_btn.click(
        fn=generate, inputs=[source_img, text, select_data], outputs=output_text
    )
gr.close_all()
demo.launch()
