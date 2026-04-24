import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from tqdm import tqdm

model_id = "vidore/colqwen2.5-v0.2"

model = ColQwen2_5.from_pretrained(
    model_id, dtype=torch.float16, device_map="auto",
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    model_id,
    use_fast=True
)

from pdf2image import convert_from_path

pdf_path = "00zentai.pdf"
images = convert_from_path(pdf_path)

embedding = []
batch_size = 2
with torch.inference_mode():
    for i in tqdm(range(0, len(images), batch_size)):
        imgs = images[i : i + batch_size]
        bf_dic = processor.process_images(imgs)
        bf_dic = bf_dic.to(model.device)
        emb = model(**bf_dic)
        emb = emb.to(dtype=torch.bfloat16)
        embedding += emb

from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

torch.save(embedding, str(data_dir / "embedding.pt"))
