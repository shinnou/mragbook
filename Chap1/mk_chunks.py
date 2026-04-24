import gzip
from pathlib import Path

out_dir = Path("data")

with gzip.open(out_dir / "ibaraki.txt.gz", 'rb') as f:
    ibaraki = f.read().decode('utf-8')

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", ""],
)

chunks = splitter.split_text(ibaraki)

import pickle

with open(out_dir / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
