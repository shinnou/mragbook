import json

from collections import defaultdict
import types

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem


def safe_json_value(x):
    """
    JSONに入れて安全な値へ変換。
    - method/function/callable は None にする
    - それ以外は str にできれば str にする
    """
    if x is None:
        return None
    # method / function / callable を弾く
    if isinstance(x, (types.FunctionType, types.MethodType)) or callable(x):
        return None
    # 基本型はそのまま
    if isinstance(x, (str, int, float, bool)):
        return x
    # list/dict は再帰で安全化
    if isinstance(x, list):
        return [safe_json_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe_json_value(v) for k, v in x.items()}
    # それ以外は文字列へ
    try:
        s = str(x)
        return s if s.strip() else None
    except Exception:
        return None

def get_caption_text(element):
    """
    Doclingのバージョン差を吸収して caption を取得する。
    - caption_text が property ならその値
    - caption_text が method なら呼び出し
    - caption / caption_text / captionなど色々試す
    """
    # よくある候補を順に試す
    candidates = ["caption_text", "caption", "captionText", "title"]
    for name in candidates:
        v = getattr(element, name, None)
        if v is None:
            continue
        # methodなら呼ぶ
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        v = safe_json_value(v)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _page_nos_from_item(item):
    prov = getattr(item, "prov", None)
    if not prov:
        return []
    if isinstance(prov, list):
        pnos = []
        for p in prov:
            pn = getattr(p, "page_no", None)
            if pn is not None:
                pnos.append(int(pn))
        return pnos
    else:
        pn = getattr(prov, "page_no", None)
        return [int(pn)] if pn is not None else []

def _text_from_item(item):
    for attr in ("text", "orig"):
        v = getattr(item, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def json_default(o):
    # 最後の保険：json.dumpsで落ちないように
    if callable(o):
        return None
    try:
        return str(o)
    except Exception:
        return None

#########################################################################def main():

import sys
from pathlib import Path

# input_pdf = Path("kousatu-1a.pdf")  # ←ここを対象PDFに
input_pdf = Path(sys.argv[1])
out_dir = Path("out_docling")
out_dir.mkdir(parents=True, exist_ok=True)


from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

from docling.document_converter import DocumentConverter, PdfFormatOption

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

conv_res = converter.convert(input_pdf)
doc = conv_res.document

pages_meta = {}

# ページ画像を保存
for _key, page in doc.pages.items():
    page_no = int(page.page_no)
    page_dir = out_dir / f"page_{page_no:04d}"
    (page_dir / "images").mkdir(parents=True, exist_ok=True)
    (page_dir / "tables").mkdir(parents=True, exist_ok=True)

    page_png = page_dir / "page.png"
    # page.image が PIL を持つケースに対応
    try:
        page.image.pil_image.save(page_png, format="PNG")
    except Exception:
        # 念のため fallback
        page.image.save(page_png, format="PNG")

    pages_meta[page_no] = {
        "page_no": page_no,
        "page_image": str(page_png.relative_to(out_dir)),
        "texts": [],
        "pictures": [],
        "tables": [],
    }

picture_counter = defaultdict(int)
table_counter = defaultdict(int)

for element, _level in doc.iterate_items():
    pnos = _page_nos_from_item(element)
    if not pnos:
        continue
    page_no = pnos[0]
    if page_no not in pages_meta:
        continue

    page_dir = out_dir / f"page_{page_no:04d}"

    # 表
    if isinstance(element, TableItem):
        table_counter[page_no] += 1
        ix = table_counter[page_no]

        try:
            df = element.export_to_dataframe(doc=doc)
            csv_path = page_dir / "tables" / f"table_{ix:02d}.csv"
            df.to_csv(csv_path, index=False)
        except Exception:
            csv_path = None

        # 表画像も取れれば保存
        img_path = page_dir / "images" / f"table_{ix:02d}.png"
        saved_img = False
        try:
            element.get_image(doc).save(img_path, "PNG")
            saved_img = True
        except Exception:
            saved_img = False

        pages_meta[page_no]["tables"].append({
            "index": ix,
            "csv": str(csv_path.relative_to(out_dir)) if csv_path else None,
            "image": str(img_path.relative_to(out_dir)) if saved_img else None,
        })
        continue

    # 図（画像）
    if isinstance(element, PictureItem):
        picture_counter[page_no] += 1
        ix = picture_counter[page_no]

        img_path = page_dir / "images" / f"picture_{ix:02d}.png"
        try:
            element.get_image(doc).save(img_path, "PNG")
            img_rel = str(img_path.relative_to(out_dir))
        except Exception:
            img_rel = None

        cap = get_caption_text(element)

        pages_meta[page_no]["pictures"].append({
            "index": ix,
            "image": img_rel,
            "caption": cap,  # ←必ずJSON安全（methodは入らない）
        })
        continue

    # テキスト
    text = _text_from_item(element)
    if text:
        label = safe_json_value(getattr(element, "label", None))
        pages_meta[page_no]["texts"].append({
            "label": label,
            "text": text,
        })

# ページごとにテキストを保存
for page_no, meta in pages_meta.items():
    page_dir = out_dir / f"page_{page_no:04d}"
    txt_path = page_dir / "text.txt"
    joined = "\n".join([t["text"] for t in meta["texts"]])
    txt_path.write_text(joined, encoding="utf-8")
    meta["text_file"] = str(txt_path.relative_to(out_dir))

    # 念のため、meta 全体を JSON 安全化
    pages_meta[page_no] = safe_json_value(meta)

payload = {
    "document": input_pdf.name,
    "pages": [pages_meta[k] for k in sorted(pages_meta)],
}

(out_dir / "pages.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
    encoding="utf-8",
)
