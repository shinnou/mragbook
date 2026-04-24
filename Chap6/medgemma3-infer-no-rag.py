#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText


prompt  = """添付された胸部X線画像を読影し、診断レポートを出力してください。"""

path = "iu_xray/images/CXR3486_IM-1695/0.png"

image = Image.open(path).convert("RGB")

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(model_id)
model.to('cuda').eval()
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "あなたは放射線科医のアシスタントです。"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# 入力をモデルのデバイスへ
inputs = {k: v.to(model.device) for k, v in inputs.items()}

gen_kwargs = dict(
    max_new_tokens=1000,
#    do_sample=(args.temperature > 0),
#    temperature=0.0,
)

input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
#    out = model.generate(**inputs, **gen_kwargs)
        out = model.generate(**inputs, max_new_tokens=1000)

# プロンプト部分を除いてデコード
gen = out[0][input_len:]
text = processor.decode(gen, skip_special_tokens=True)

print(text)
