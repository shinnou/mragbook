import os
from pathlib import Path

root_path = Path("out_docling")

corpus = ""
chunks = []
image_files = []
for page_dir in sorted(root_path.glob("page_*")):
    text_file = page_dir / "text.txt"
    images_dir = page_dir / "images"
    with open(text_file, 'r', encoding='utf-8') as file:
        page_text = file.read()
    corpus += page_text
    image_files += list(images_dir.glob("picture_*.png")) + list(images_dir.glob("table_*.png"))

import pickle

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "image_files.pkl", "wb") as f:
    pickle.dump(image_files, f)


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", ""],
)

chunks += splitter.split_text(corpus)

with open(out_dir / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
