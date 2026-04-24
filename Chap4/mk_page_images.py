from pdf2image import convert_from_path

pdf_path = "00zentai.pdf"
images = convert_from_path(pdf_path)

from pathlib import Path

doc_dir = Path("doc")
doc_dir.mkdir(parents=True, exist_ok=True)

# doc ディレクトリに各ページ画像を保存する
for i, image in enumerate(images):
    filename = f"page_{i+1}.png"
    image.save(doc_dir / filename, "PNG")
