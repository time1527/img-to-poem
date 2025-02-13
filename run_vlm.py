import os
import sys
import gradio as gr

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)
from vlm import generate_poem


with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        gr.Markdown("""<h1><center>ImagetoPoem</center></h1>""")
    with gr.Row():
        source_img = gr.Image(
            type="filepath", label="Upload Image", height=600, width=600
        )
    with gr.Row():
        gen_btn = gr.Button("Generate Poem")
    with gr.Row():
        output_text = gr.Textbox(label="Output", scale=2)

    gen_btn.click(fn=generate_poem, inputs=[source_img], outputs=output_text)
gr.close_all()
demo.launch()
