#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# import os
# from typing import Optional

import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText

template = """
添付された胸部X線画像を読影し、診断レポートを書いて下さい。また添付された画像と類似する画像に対する診断レポートを付けます。この類似画像の診断レポートも参考にして、診断レポートを書いて下さい。注意として類似画像の診断レポートには誤りが含まれているかもしれないので、そのままコピーすることは絶対にやめてください。添付された胸部X線画像を必ず読影して、その診断レポートを書いて下さい。類似画像の診断レポートはあくまで参考として、利用してください。

【類似画像の診断レポート】
{ref_report}
"""

ref_report = ... # top-1 類似画像の診断レポート

# temp = """あなたは放射線科医のアシスタントです。添付された胸部X線画像を読影し、診断レポートを出力してください。
# ただし、これは臨床診断の代替ではない「下書き」です。断定を避け、不確実性があれば明示し、必要なら追加検査や臨床情報の確認を提案してください。
# また入力された画像と類似画像の診断レポートも参考にしてください。類似画像の診断レポートには誤りが含まれているかもしれないので、そのままコピーすることはやめてください。類似画像の診断レポートはあくまで参考として利用してください。

# 【類似画像の診断レポート】
# {sankou}
# """

path = "iu_xray/images/CXR3486_IM-1695/0.png"

ref_report  = """
心臓は正常な大きさである。縦隔に異常所見はない。胸水、気胸、または限局性気道疾患は認められない。脊椎に軽度の慢性変性変化が認められる。
"""

prompt = template.format(ref_report=ref_report)

image = Image.open(path).convert("RGB")

# model_id = "google/medgemma-27b-it"
model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(model_id)
model.to('cuda').eval()
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "あなたは放射線科の優秀な医者です。"}],
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
    out = model.generate(**inputs, **gen_kwargs)

# プロンプト部分を除いてデコード
gen = out[0][input_len:]
text = processor.decode(gen, skip_special_tokens=True)

print(text)
