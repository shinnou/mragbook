import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip2-so400m-patch14-384"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

if device == "cuda":
    model = model.to(torch.float16)

def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


@torch.inference_mode()
def embding_batch_text(chunks: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(chunks), batch_size):
        inputs = processor(
            text=chunks[i : i + batch_size],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        text_features = model.get_text_features(**inputs)
        #  v = text_features.detach().float().cpu().numpy()  # (B, D)
        v = text_features.pooler_output.detach().float().cpu().numpy()        
        vecs.append(v)
    return normalize(np.vstack(vecs).astype(np.float32))


@torch.inference_mode()
def embding_batch_image(image_paths: list[str], batch_size: int = 16) -> np.ndarray:
    vecs = []
    for i in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        img_features = model.get_image_features(**inputs)
        # v = img_features.detach().float().cpu().numpy()  # (B, D)
        v = img_features.pooler_output.detach().float().cpu().numpy()  # (B, D)
        vecs.append(v)
    return normalize(np.vstack(vecs).astype(np.float32))


#############################################################

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


import pickle

with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

text_vecs  = embding_batch_text(chunks, 3)
text_index = build_faiss_index(text_vecs)

faiss.write_index(text_index, "data/text_index.faiss")

#---------------------

with open("data/image_files.pkl", "rb") as f:
    image_files = pickle.load(f)

image_vecs = embding_batch_image(image_files, 3)
image_index = build_faiss_index(image_vecs)

faiss.write_index(image_index,"data/image_index.faiss")
