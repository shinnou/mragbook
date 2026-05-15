from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoProcessor
)
from PIL import Image
import requests
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     vlm_model_id, dtype="auto", device_map="auto"
# ).eval()

vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_model_id, dtype="auto", device_map="auto", offload_buffers=True,
).eval()

# vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, use_fast=True)   ## .to(device)
vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)   ## .to(device)


system_prompt = """
あなたはマルチモーダルRAGシステムの回答生成アシスタントです。

ユーザの質問に対して、添付された画像と「検索で得られたテキストチャンク」を参考にして、できるだけ丁寧に回答してください。

添付された画像や「検索で得られたテキストチャンク」では答えが分からない場合は、必ずその旨を述べて、考えられる回答を答えてください。
"""

user_prompt_template = """
「タスク」に記載されている指示に従って、ユーザの質問にできるだけ丁寧に回答して下さい。

### ユーザの質問
{query}

### 検索で得られたテキストチャンク
{chunk_text}

### タスク
添付された画像と上記の「検索で得られたテキストチャンク」を参考にして「ユーザの質問」に回答してください。

ただし以下の指示に従ってください。
1. 根拠（テキスト・画像）に含まれない内容を推測で断定しない。
2. 画像が質問と直接関係ない場合は、無理に参考にせず、テキストチャンクを優先して参考にする。
3. 画像を用いた説明では、画像のどの部分に基づくかを言語化する。
4. 回答は日本語で、丁寧に正確に述べる。
"""

class MM_RAG:
    def __init__(self, model, processor, faiss_mm_db, system_prompt, user_prompt_template):
        self.model = model
        self.processor = processor
        self.faiss_mm_db = faiss_mm_db
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    @torch.inference_mode()
    def generate(self, query):
        chunk_text = self.faiss_mm_db.text_retrieve(query)
        image_file = self.faiss_mm_db.image_retrieve(query)
        print(image_file)
        user_prompt = user_prompt_template.format(
            query=query,
            chunk_text=chunk_text,
        )
        messages = [
            {
               "role": "system",
               "content": [
                   {"type": "text", "text": system_prompt}
               ]
            },
            {
               "role": "user",
               "content": [
                   {"type": "image", "image": image_file},
                   {"type": "text", "text": user_prompt}
               ]
            }

        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = len(inputs["input_ids"][0])
        generation = self.model.generate(
              **inputs,
              max_new_tokens=200,
              do_sample=False
        )
        answer_ids = generation[0][input_len:]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True)
        return answer

#######


class FAISS_MM_DB:
    def __init__(self, model, processor, text_index, chunks, image_index, image_files, k=3):
        self.model = model
        self.processor = processor
        self.text_index = text_index
        self.chunks = chunks
        self.image_index = image_index
        self.image_files = image_files
        self.k = k

    @torch.inference_mode()
    def embed_query_text(self, query: str):
        # SigLIP2 では attention_mask が無い場合があるので安全に
        device = self.model.device
        inputs = self.processor(
            text=[query],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        if "attention_mask" in inputs:
            attn = inputs["attention_mask"].to(device)
            v = self.model.get_text_features(input_ids=input_ids, attention_mask=attn)
        else:
            v = self.model.get_text_features(input_ids=input_ids)
        # v = v.detach().float().cpu().numpy().astype(np.float32)  # (1, D)
        v = v.pooler_output.detach().float().cpu().numpy().astype(np.float32)  # (1, D)            
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return v

    def text_retrieve(self, query: str):
        # 1. E5モデルのルールに従い、クエリ用プレフィックスを付与
        query_with_prefix = f"query: {query}"
        # 2. 質問のベクトル化
        query_vector = self.embed_query_text(query_with_prefix)
        # 3. FAISS 検索の実行
        # distances: 距離, indices: chunksのインデックス番号
        distances, indices = self.text_index.search(
            query_vector.astype('float32'),
            self.k
        )
        # 4. 結果の整形 (インデックス番号を実際のテキストに変換)
        # ここでは top の１つだけを返すことにする
        return self.chunks[0]

    def image_retrieve(self, query: str):
        query_vector = self.embed_query_text(query)
        # 3. FAISS 検索の実行
        # distances: 距離, indices: chunksのインデックス番号
        distances, indices = self.image_index.search(
            query_vector.astype('float32'),
            self.k
        )
        # 4. 結果の整形 (インデックス番号を実際のテキストに変換)
        # ここでは top の１つだけを返すことにする
        return str(self.image_files[0])

#---------------------------

import os
import numpy as np

## クロスモーダル埋め込みモデルとそのプロセッサの読み込み

cembd_model_id = "google/siglip2-so400m-patch14-384"
device = "cuda" if torch.cuda.is_available() else "cpu"

cembd_processor = AutoProcessor.from_pretrained(cembd_model_id)
cembd_model = AutoModel.from_pretrained(cembd_model_id).to(device).eval()

## VLM とそのプロセッサの読み込み

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoProcessor
)
from PIL import Image
import requests
import torch

vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_model_id, dtype="auto", device_map="auto"
).eval()

# vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, use_fast=True)   ## .to(device)
vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)   ## .to(device)

if device == "cuda":
    cembd_model = cembd_model.to(torch.float16)

# チャンクのデータベースとその faiss インデックスファイルの読み込み

import pickle
import faiss

with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

text_index = faiss.read_index("data/text_index.faiss")

# 画像のデータベースとその faiss インデックスファイルの読み込み

with open("data/image_files.pkl", "rb") as f:
    image_files = pickle.load(f)

image_index = faiss.read_index("data/image_index.faiss")



faiss_mm_db = FAISS_MM_DB(cembd_model, cembd_processor, text_index, chunks, image_index, image_files, k=3)

query = "LINE で写真を送るには、どの画面で、どのボタンを押せばよいですか"

mm_rag = MM_RAG(vlm_model, vlm_processor, faiss_mm_db, system_prompt, user_prompt_template)
answer = mm_rag.generate(query)
print(answer)
