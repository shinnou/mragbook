from datasets import load_dataset

wikija_dataset = load_dataset(
    path="singletongue/wikipedia-utils",
    name="passages-c400-jawiki-20230403",
    split="train",
)

ibaraki = ""
tstr = '茨城'
for data in wikija_dataset:
    if ((tstr in data['title']) or (tstr in data['text'])):
        ibaraki += (data['text'] + "\n\n")

import gzip
from pathlib import Path

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

with gzip.open(out_dir / "ibaraki.txt.gz", 'wb') as f:
    f.write(ibaraki.encode('utf-8'))

