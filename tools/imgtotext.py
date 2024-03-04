import sys
sys.path.append("../")

import torch
from PIL import Image
from LOCALPATH import CV_PATH
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_from_img(img,LLM):
    """
    https://huggingface.co/Salesforce/blip-image-captioning-base
    img:path
    """
    
    processor = BlipProcessor.from_pretrained(CV_PATH)
    cv_model = BlipForConditionalGeneration.from_pretrained(CV_PATH).to(DEVICE).eval()
    raw_image = Image.open(img).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)
    out = cv_model.generate(**inputs,max_new_tokens=200)
    text = processor.decode(out[0], skip_special_tokens=True)

    prompt = f"""将文本翻译为中文：{text}"""
    response = LLM(prompt)
    return response