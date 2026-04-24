# import os
import torch
# from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

query = (
   "違法あるいは有害な情報に関する投稿の目撃経験として"
   "最も多いものと、最も少ないものは、それぞれどのような"
   "投稿ですか？"
)

repo_id = "vidore/colqwen2.5-v0.2"

model = ColQwen2_5.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="auto" ,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    repo_id,
    use_fast=True
)

p_query = processor.process_queries([ query ])
p_query.to(model.device)

with torch.inference_mode():
    query_embedding = model(**p_query)

query_embedding = query_embedding.to(dtype=torch.bfloat16)


emb_path ="data/embedding.pt"
embedding = torch.load(emb_path)

scores = processor.score_multi_vector(query_embedding, embedding)[0]

values, indices = torch.topk(scores, k=3)

for i in range(len(indices)):
    print(f"ページ: {indices[i] + 1} スコア: {values[i]}")

img_file = f"doc/page_{indices[0] + 1}.png"

#--------------------------------------------------------------------------------------

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import requests


vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_model_id, dtype="auto", device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(vlm_model_id, use_fast=True)

system_prompt = """
あなたは技術文書や統計的な調査報告書を理解を支援する専門アシスタントです。
"""
user_prompt_template = """
添付の画像は、ある PDF 文書の１ページです。そのページにある画像も含めて、このページの内容を理解し、ユーザからの質問に、できるだけ丁寧に、かつ正確に、回答して下さい。

### ユーザからの質問
{query}
"""

user_prompt = user_prompt_template.format(query=query)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_file},
            {"type": "text", "text": user_prompt}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = len(inputs["input_ids"][0])

with torch.inference_mode():
    generation = vlm_model.generate(**inputs, max_new_tokens=2000, do_sample=False)

ans_ids = generation[0][input_len:]
answer = processor.decode(ans_ids, skip_special_tokens=True)
print(answer)
