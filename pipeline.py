import fitz
import base64
import io
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
from prompts import FIELD_EXTRACTION_PROMPT, PRESENCE_PROMPT, SUMMARY_PROMPT

client = OpenAI(api_key="YOUR_API_KEY")

# ---------------- PDF → Images ----------------

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num + 1, img))

    return images


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------- GPT FIELD EXTRACTION ----------------

def extract_fields(image, page_number):
    base64_img = encode_image(image)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FIELD_EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Page number: {page_number}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


# ---------------- SIGNATURE PRESENCE ----------------

def check_presence(cropped_image):
    base64_img = encode_image(cropped_image)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PRESENCE_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


# ---------------- EMBEDDINGS ----------------

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_embedding(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(tensor)
    return emb.flatten().numpy()


def compare_signatures(img1, img2):
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    return cosine_similarity([emb1], [emb2])[0][0]


# ---------------- CONFIDENCE ----------------

def compute_confidence(field_conf, presence_conf, identity_conf):
    return 0.4*np.mean(field_conf) + 0.3*np.mean(presence_conf) + 0.3*np.mean(identity_conf)
