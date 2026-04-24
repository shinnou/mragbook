import numpy as np

#############################

class FAISS_DB:
    def __init__(self, model, index, chunks, k=3):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.k = k
    def retrieve(self, query: str):
        # 1. E5モデルのルールに従い、クエリ用プレフィックスを付与
        query_with_prefix = f"query: {query}"
        # 2. 質問のベクトル化
        query_vector = self.model.encode([query_with_prefix])
        # 3. FAISS 検索の実行
        # distances: 距離, indices: chunksのインデックス番号
        distances, indices = self.index.search(
            query_vector.astype('float32'),
            self.k
        )
        # 4. 結果の整形 (インデックス番号を実際のテキストに変換)
        results = []
        for idx in indices[0]:
            if idx != -1:  # 該当なし (-1) を除外
                results.append(self.chunks[idx])
        return results


#############################

from sentence_transformers import SentenceTransformer
model_name = 'intfloat/multilingual-e5-large'
model = SentenceTransformer(model_name)

import faiss
# index = faiss.read_index("ibaraki_index.faiss")
index = faiss.read_index("kousatu-1_index.faiss")

import pickle
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

####################

# すでに定義済みの model, index, chunks を渡してインスタンス化
db = FAISS_DB(model, index, chunks, k=3)

# 検索の実行
query = "日本、米国、中国、韓国の中で、SNS の利用率が最も高い国と低い国をどこですか？"
related_chunks = db.retrieve(query)

# 結果の確認
for i, text in enumerate(related_chunks):
    print(f"関連文章 {i+1}: {text}")
