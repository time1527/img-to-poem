import base64
import requests
from urllib.parse import urlparse


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_image_url(image):
    if not is_url(image):
        base64_image = encode_image(image)
        return f"data:image/jpeg;base64,{base64_image}"
    return image


def get_image_response(url, prompt):
    response = requests.post(
        "https://text.pollinations.ai/openai",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            "model": "openai",
        },
    )
    return response.json()["choices"][0]["message"]["content"]


def image_caption(image):
    url = get_image_url(image)
    prompt = "What's in this image? Answer in Chinese."
    return get_image_response(url, prompt)


def generate_poem(image):
    url = get_image_url(image)
    prompt = "Write a line of ancient poetry about the image. Answer in Chinese."
    return get_image_response(url, prompt)


# if __name__ == "__main__":
#     # print(image_caption("/home/pika/Downloads/F31g75LXkAAVdEk.jpeg"))

#     print(
#         generate_poem(
#             "https://images.unsplash.com/photo-1709146878535-b1b3f1374642?q=80&w=5096&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
#         )
#     )
