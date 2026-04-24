import requests
import io
from PyPDF2 import PdfReader, PdfWriter

url = "https://www.niye.go.jp/wp-content/uploads/2024/07/zentai-1.pdf"
response = requests.get(url)
response.raise_for_status()

pdf_stream = io.BytesIO(response.content)
reader = PdfReader(pdf_stream)
writer = PdfWriter()

start_page = 57
end_page = 65

for page_num in range(start_page - 1, end_page):
    writer.add_page(reader.pages[page_num])

from pathlib import Path

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)
    
with open(out_dir / "kousatu-1.pdf", "wb") as output_file:
    writer.write(output_file)
