import torch
from PIL import Image
# from open_clip import create_model_from_pretrained, get_tokenizer
from open_clip import create_model_from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

model, preprocess = create_model_from_pretrained(model_id)
# tokenizer = get_tokenizer(model_id)

model = model.to(device).eval()


import numpy as np
import faiss

def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    """
    vecs: (N, D) float32, L2正規化済み前提
    cosine検索したいので IndexFlatIP を使う（内積＝cosine）
    """
    vecs = vecs.astype(np.float32)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index

def build_faiss_index(vecs):
    """
    vecs: (N, D) float32, L2正規化済み前提
    cosine検索したいので IndexFlatIP を使う（内積＝cosine）
    """
    vecs = vecs.astype(np.float32)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index

#---------------------------

import json

with open('iu_xray/annotation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

features = []
data_report = []
data_image = []
with torch.inference_mode():
    for ele in data['train']:
        id = ele['id']
        rep = ele['report']
        imgs = ele['image_path']
        for img in imgs:
            img = "iu_xray/images/" + img
            data_image.append(img)
            img = preprocess(Image.open(img)).unsqueeze(0).to(device)
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True) # 正規化
            features.append(feat.cpu().numpy())
            data_report.append(rep)


import pickle

img_index = build_faiss_index(np.vstack(features))

from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

faiss.write_index(img_index, str(data_dir / "img_index.faiss"))

with open(str(data_dir / "data_image.pkl"), "wb") as f:
    pickle.dump(data_image, f)

with open(str(data_dir / "data_report.pkl"), "wb") as f:
    pickle.dump(data_report, f)
