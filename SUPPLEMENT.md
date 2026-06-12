# 補足説明

## 2章で使うモデル

2章で使うモデルは書籍では gpt-4.1-mini か gpt-4.1 と書いていましたが、現在は gpt-5.4-mini か gpt-5.4 も使えますし、こちらの方が良い結果がでる感じです。

## ColQwen3.5

4章では ColQwen2_5 を使っていましたが、ColQwen3.5 があります。まだ確認できていませんが、多分、使えると思います。cuda 13 でも大丈夫かもしれません。

https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3


ColQwen3.5 を動かすための transformers 5.x と普通に入っている vllm 0.19.0 は共存できないので、vllm を含まない環境を構築した方が良いです。以下で構築した colqwen_pure の環境で動きます。

conda create -n colqwen_pure python=3.11 -y
conda activate colqwen_pure
pip install torch torchvision torchaudio
pip install colpali-engine






