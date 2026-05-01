# マルチモーダル RAG 入門

本リポジトリは、書籍「マルチモーダル RAG 入門」に掲載しているサンプルプログラムを Chapter 別に収録したものです。

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

**GPU がない場合**

```bash
pip3 install torch torchvision
```

PyTorch のインストール後、各 Chapter ディレクトリの `README.md` に従って残りの依存ライブラリをインストールしてください。
