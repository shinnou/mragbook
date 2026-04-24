本章で解説したプログラムを動かすための実行手順を示しておきます。ライブラリのインストールは全て済んでいることを仮定しているので、不足のライブラリなどがあれば、その都度、インストールして下さい。

(1) サンプルで利用する PDF 文書をダウンロードする

```bash
$ curl -O https://www.soumu.go.jp/johotsusintokei/\
whitepaper/ja/r07/pdf/00zentai.pdf
```

> 00zentai.pdf がダウンロードされる

(2) PDF 文書をページに分割し、各ページを画像ファイルに変換する

```bash
$ pip install pdf2image
$ python mk_page_images.py
```

> `doc` というディレクトリが作成され、その下に各ページが PNG 画像として保存される

(3) ページ画像から ColPali 用のインデックスを作成する

```bash
$ pip install colpali_engine
$ python create_embedding.py
```

> data/embedding.pt が作成される

(4) RAG を実行、クエリはプログラム内で設定

```bash
$ python mmrag_colpali.py
```

> *** 回答が表示される ***
