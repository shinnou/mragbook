本章で解説したプログラムを動かすための実行手順を示しておきます。ライブラリのインストールは全て済んでいることを仮定しているので、不足のライブラリなどがあれば、その都度、インストールして下さい。

---

(1) データベースの元データとなるテキストデータの構築

```bash
$ python mk_base_text_data.py
```

> data/ibaraki.txt.gz が作成される

(2) テキストデータをチャンクの集合に分割

```bash
python mk_chunks.py
```

> data/chunks.pkl が作成される

(3) チャンクの集合から faiss のインデックスを作成

```bash
python mk_text_index.py
```

> data/ibaraki_index.faiss が作成される

(4) 検索器を作り、SimpleRetrievalQA のインスタンスを作り、RAG 実行。クエリはプログラム内で設定

```bash
python naive_rag.py
```

> *** 回答が表示される ***
