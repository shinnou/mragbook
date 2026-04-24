from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
from PIL import Image
import requests
import torch

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, dtype="auto", device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

system_prompt = """
あなたは技術文書や統計的な調査報告書を理解を支援する専門アシスタントです。

あなたの役割は、文書中に含まれる図・グラフ・表・構成図などを、与えられた画像と文書テキストの情報のみに基づいて、正確かつ忠実に文章で説明することです。

与えられていない情報を推測したり補完したりしてはいけません。不明確な点がある場合は、その旨が分かるように慎重に記述してください。
"""

user_prompt_template = """
添付の画像は、ある PDF 文書の 1 ページ分のテキスト内にある画像です。「ページ本文」はそのページのテキストです。この「ページ本文」の情報を参照して、「タスク」に記載されている指示を実行し、「出力」に記載されている形で出力してください。

### ページ本文
{page_text}

### タスク
上記のページ本文を文脈として参考にしながら、与えられた画像が何を示しているのかを自然言語で正確に説明してください。

以下の指示に従ってください。
1. まず、この画像の種類を特定してください。（例：グラフ、表、構成図、フローチャート、概念図など）
2. 画像が何を測定・比較・説明しているのかを、画像の内容とページ本文の情報を踏まえて説明してください。
3. 画像がグラフの場合：
   - 横軸と縦軸がそれぞれ何を表しているかを説明してください。
   - 主な傾向、差異、特徴的な点を文章でまとめてください。
4. 画像が表の場合：
   - 行と列がそれぞれ何を表しているかを説明してください。
   - 表から読み取れる重要な結果や比較を要約してください。
5. 画像が構成図・概念図・フローチャートの場合：
   - 主な要素と、それらの関係や流れを説明してください。
6. 「この図」「この画像」といった参照表現は使わず、
   その説明文単体を読んだだけで内容が理解できる文章にしてください。
7. 画像や本文から読み取れない数値・意味・ラベルを
   勝手に補ったり推測したりしてはいけません。

### 出力
検索拡張生成（RAG）の知識データとしてそのまま利用できるような、簡潔で情報量のある説明文を書いてください。
"""

class Generate_Imgae_Text:
    def __init__(self, model, processor, system_prompt, user_prompt_template):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
    def generate(self, text_file, image_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            page_text = file.read()
        user_prompt = user_prompt_template.format(page_text=page_text)
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
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = len(inputs["input_ids"][0])
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=False
            )
        answer_ids = generation[0][input_len:]
        answer = processor.decode(answer_ids, skip_special_tokens=True)
        return answer

import os
from pathlib import Path

gen_img_text = Generate_Imgae_Text(model, processor, system_prompt, user_prompt_template)

root_path = Path("out_docling")

for page_dir in root_path.glob("page_*"):
    text_file = page_dir / "text.txt"
    images_dir = page_dir / "images"
    # 一応、text.txt と imagesディレクトリが存在することを確認
    if text_file.exists() and images_dir.is_dir():
        image_files = list(images_dir.glob("picture_*.png")) + list(images_dir.glob("table_*.png"))
        for image_path in image_files:
            text = gen_img_text.generate(str(text_file), str(image_path))
            output_file_path = image_path.with_suffix(".txt")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  - Saved: {output_file_path.name}")
