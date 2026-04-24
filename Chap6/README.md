本章で解説したプログラムを動かすための実行手順を示しておきます。ライブラリのインストールは全て済んでいることを仮定しているので、不足のライブラリなどがあれば、その都度、インストールして下さい。

(1) IU X-Ray データセットを準備する

ブラウザー経由で以下から `iu_xray.zip` をダウンロードし、解凍する。

```
https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view
```

(2) IU X-Ray データセットの train データから RAG のデータベースを作成する

```bash
$ pip install open_clip_torch==2.23.0 pillow
$ python mk_database.py
```

> data/img_index.faiss と data/data_report.pkl が作成される。これがデータベースとなる。

(3) 1番目の test データの類似画像とそのレポートを検索してみる

```bash
$ python search_report.py
```

> 検索された画像のファイル名とレポートが表示される

(4) (3)で得られた結果をプログラム内で利用して、1番目の test データのレポートを作成する

```bash
$ python medgemma3-infer.py
```

> *** レポートが表示される ***
