
#############################

import numpy as np

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
embd_model_name = 'intfloat/multilingual-e5-large'
embd_model = SentenceTransformer(embd_model_name)

import faiss
index = faiss.read_index("data/kousatu-1_index.faiss")

import pickle
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

####################

database = FAISS_DB(embd_model, index, chunks, k=3)

####################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. モデルとトークナイザーの準備
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
).eval()

##--------------------------------------------

from transformers import GenerationConfig

class SimpleRetrievalQA:
    def __init__(self, model, tokenizer, database, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.database = database
        self.prompt = prompt
    def mk_input(self, query, related_chunks, k):
        context_str = "\n\n".join(related_chunks[:k])
        user_content = self.prompt.format(context_str=context_str, query=query)
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                   ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return inputs
    @torch.inference_mode()
    def invoke(self, query):
        related_chunks = self.database.retrieve(query)
        inputs = self.mk_input(query, related_chunks, self.database.k)
        config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=512
        )
        generated_ids = model.generate(**inputs, generation_config=config)
        # 生成されたID（output_ids）から入力部分の長さを削る
        generated_ids = generated_ids[0][len(inputs.input_ids[0]):]
        # デコードして文字列にする
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

# ####################

system_prompt = """あなたは提供された「文脈」のみを使用して質問に答えるAIアシスタントです。
以下のルールを厳守してください：
1. 「文脈」に含まれていない情報は絶対に回答に含めないでください。
2. 自分の知識や一般常識を使用しないでください。
3. 「文脈」から回答が見つからない場合は、正直に「提供された情報からは回答できません」と答えてください。
4. 回答は簡潔かつ具体的に記述してください。"""

prompt = """以下の「文脈」を注意深く読み、質問に答えてください。

### 文脈:
{context_str}

### 質問:
{query}
"""

rag = SimpleRetrievalQA(model, tokenizer, database, prompt)

query = "日本、米国、中国、韓国の中で、SNS の利用率が最も高い国と低い国をどこですか？"

result = rag.invoke(query)
print(result)
