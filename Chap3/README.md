本章で解説したプログラムを動かすための実行手順を示しておきます。ライブラリのインストールは全て済んでいることを仮定しているので、不足のライブラリなどがあれば、その都度、インストールして下さい。

## 画像をテキスト化するアプローチ

(1) サンプルで利用する PDF 文書をダウンロードし、対象となるページ部分だけ取り出す

```bash
$ pip install PyPDF2
$ python extact_pages.py
```

> data/kousatu-1.pdf が作成される

(2) (1)で作成した PDF 文書から Docling によりテキストと画像を切り分けて抽出する

```bash
$ pip install docling
$ python extract_docling_per_page.py data/kousatu-1.pdf
```

> `out_docling` というディレクトリが作成され、各ページの情報が保存される

(3) 文書内の各画像をテキスト化し、それをファイルに保存

```bash
$ python save_image_text.py
```

> 各ページ内にある画像ファイルがテキスト化され、ファイルとして保存される

(4) 文書内のテキストと画像のテキストを合わせてチャンクに分割して保存する

```bash
$ python mk_chunks.py
```

> data/chunks.pkl が作成される

(5) (4) で作成されたチャンクのデータベースのインデックスを作成する

```bash
$ python mk_index.py
```

> data/kousatu-1_index.faiss が作成される

(6) RAG を実行、クエリはプログラム内で設定

```bash
$ python mm_rag_image_to_text.py
```

> *** 回答が表示される ***

## 画像とテキストを同一空間に埋め込むアプローチ

(1) サンプルで利用する PDF 文書をダウンロードする

```bash
$ curl -O https://www.city.fukushima.fukushima.jp/\
material/files/group/6/sumahohanndobook2025.pdf
```

> sumahohanndobook2025.pdf がダウンロードされる

(2) (1)で作成した PDF 文書から Docling によりテキストと画像を切り分けて抽出する

```bash
$ python extract_docling_per_page.py sumahohanndobook2025.pdf
```

> out_docling というディレクトリが作成され、各ページの情報が保存される

(3) PDF 文書のテキストをチャンクに分割し、また PDF 文書内の画像の集合を作成する

```bash
$ python mk_chunks_image_files.py
```

> data/chunks.pkl と data/images_files.pkl が作成される

(4) (3) で作成されたチャンクと画像をベクトル化し、データベースのインデックスを作成する

```bash
$ python mk_faiss_index.py
```

> data/text_index.faiss と data/image_index.faiss が作成される

(5) RAG を実行、クエリはプログラム内で設定

```bash
$ python mm_rag_image_to_vec.py
```

> *** 回答が表示される ***
