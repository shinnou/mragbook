import os
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

query = "違法あるいは有害な情報に関する投稿の目撃経験として、最も多いものはどのような投稿ですか？"

repo_id = "vidore/colqwen2.5-v0.2"
emb_path ="./embedding.pt"

model = ColQwen2_5.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" ,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    repo_id,
    use_fast=True
)


p_query = processor.process_queries([ query ])
p_query = p_query.to(model.device)

with torch.no_grad():
    query_embedding = model(**p_query)

embedding = torch.load(emb_path, weights_only=True)

scores = processor.score_multi_vector(query_embedding, embedding)[0]

values, indices = torch.topk(scores, k=5)

for i in range(len(indices)):
    print(f"ページ: {indices[i] + 1} スコア: {values[i]}")
