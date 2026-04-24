# pip install faiss-cpu sentence-transformers

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import pickle
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ### CPU

# model_name = 'intfloat/multilingual-e5-large'
# model = SentenceTransformer(model_name)
# passages = [f"passage: {chunk}" for chunk in chunks]
# embeddings = model.encode(passages)

### GPU

import torch
model_name = 'intfloat/multilingual-e5-large'
model = SentenceTransformer(model_name, device='cuda')

passages = [f"passage: {chunk}" for chunk in chunks]
embeddings = model.encode(passages, batch_size=64, show_progress_bar=True)

# FAISS インデックスの作成
dimension = embeddings.shape[1]  # モデルの出力次元数（e5-largeは1024）
index = faiss.IndexFlatL2(dimension)  # L2距離（ユークリッド距離）でインデックス初期化

# ベクトルをインデックスに追加
# FAISSは float32 型を期待するため変換します
index.add(embeddings.astype('float32'))

# インデックスをファイルに保存

faiss.write_index(index, "data/kousatu-1_index.faiss")
