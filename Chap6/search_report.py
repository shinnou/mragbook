from pathlib import Path

data_dir = Path("data")

import faiss

img_index = faiss.read_index(str(data_dir / "img_index.faiss"))

import pickle

with open(str(data_dir / "data_report.pkl"), "rb") as f:
    data_report = pickle.load(f)

with open(str(data_dir / "data_image.pkl"), "rb") as f:
    data_image = pickle.load(f)


import json

with open('iu_xray/annotation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ele = data['test'][0]
test_img =  "iu_xray/images/" + ele['image_path'][0]
test_report = ele['report']


import torch
from PIL import Image
from open_clip import create_model_from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, preprocess = create_model_from_pretrained(model_id)
model = model.to(device).eval()

with torch.inference_mode():
    img = preprocess(Image.open(test_img)).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    feat /= feat.norm(dim=-1, keepdim=True) # 正規化
    feat = feat.cpu().numpy()

distances, indices = img_index.search(feat.astype('float32'), k=3)

top1 = indices[0][0]
hit_image = data_image[top1]
hit_report = data_report[top1]

#----------------------------------
print("-- test ---")
print(test_img)
print(test_report)
print()
print("-- retrieve ---")
print(hit_image)
print(hit_report)
