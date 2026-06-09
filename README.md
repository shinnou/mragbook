# 入門マルチモーダルRAG

本リポジトリは、書籍『入門マルチモーダルRAG ― 図表を読み解くAIアプリケーションの実装』のサンプルプログラムを配布するためのリポジトリです。

本リポジトリでは、書籍の各章で解説したサンプルプログラム、実行手順、補足情報、正誤情報を公開しています。

## 書籍情報
- 書名：入門マルチモーダルRAG: 図表を読み解くAIアプリケーションの実装
- 著者：新納浩幸
- 出版社: オーム社
- 発売日：2026/6/17
- ISBN-10: 4274234932
- ISBN-13: 978-4274234934

## ディレクトリ構成

```
mragbook/
├── Chap1/   # Chapter 1 のサンプルプログラム
├── Chap2/   # Chapter 2 のサンプルプログラム
├── Chap3/   # Chapter 3 のサンプルプログラム
├── Chap4/   # Chapter 4 のサンプルプログラム
├── Chap5/   # Chapter 5 のサンプルプログラム
└── Chap6/   # Chapter 6 のサンプルプログラム
```

各 Chapter のディレクトリには `README.md` が置かれており、そのChapterのプログラムを動かすための手順が記載されています。プログラムを実行する前に、各ディレクトリの `README.md` を参照してください。

## サンプルプログラムの取得方法

以下のコマンドで本リポジトリを取得できます。

```bash
git clone https://github.com/shinnou/mragbook.git
cd mragbook
```

## 環境構築

### 仮想環境の作成

各 Chapter のプログラムを動かす前に、conda による仮想環境の構築を推奨します。以下のコマンドで仮想環境を作成してください。

```bash
conda create -n mragbook python==3.12
conda activate mragbook
```

### PyTorch のインストール

依存ライブラリをインストールする前に、PyTorch を以下の手順で予めインストールしておくことを推奨します。

**GPU がある場合（CUDA 12.6）**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

本書の動作確認は CUDA 12.6 環境で行っています。CUDA 13 環境では、ColQwen 関連のライブラリが正常に動作しない場合があるため、Chap4 以降のプログラムは CUDA 12.6 環境での実行を推奨します。


**GPU がない場合**

```bash
pip3 install torch torchvision
```

GPU がない環境でも一部のプログラムは実行できますが、Chap4 以降の処理では計算時間やメモリ使用量の点から GPU 環境での実行を推奨します。

PyTorch のインストール後、各 Chapter ディレクトリの `README.md` に従って残りの依存ライブラリをインストールしてください。

## 関連情報
- [正誤表](ERRATA.md)
- [補足説明](SUPPLEMENT.md)
- [更新履歴](CHANGELOG.md)


## 質問・不具合報告・正誤情報について

サンプルプログラムの不具合や、書籍本文の誤植・誤りを見つけた場合は GitHub Issues からご報告ください。以下のページからお願いします。

- [Issues](https://github.com/shinnou/mragbook/issues)
- [新しい Issue を作成](https://github.com/shinnou/mragbook/issues/new/choose)

GitHub Issues を使うには GitHub アカウントが必要です。

なお、個別の実行環境に依存する問題、GPU/CUDA の設定、外部サービスやライブラリの仕様変更に起因する問題については、すべてに対応できない場合があります。
