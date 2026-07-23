# 正誤表

・p.27 の create_db.py の open に括弧が抜けています。正しくは以下です。配布のコードは正しいです。(2026年7月23日)

```
with open("data/ibaraki_vec_db.json", "w", encoding="utf-8") as f:
```

・p.28 の generate_ans.py の open に括弧が抜けています。正しくは以下です。配布のコードは正しいです。(2026年7月23日)

```
with open("data/ibaraki_vec_db.json", "r", encoding="utf-8") as f:
```


・p.10 ファイル名が mk-rag-db-from-text.py となっているが、mk_text_index.py の間違い (2026年7月23日)

・p.11 ファイル名（3ヶ所）が mk_retriver.py の retriver は retriever のタイポ。ただし実際に配布しているファイル名も mk_retriver.py だったで、配布のファイル名も mk_retriever.py に修正 (2026年7月23日)

・p.19 の SimpleRetrievalQA クラスの定義が配布のプログラムと大きく違います。これは別のコードを書籍の方に写したのが原因です。配布のプログラム（以下のコード）が正しいです。

```
class SimpleRetrievalQA:
    def __init__(self, model, tokenizer, database, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.database = database
        self.prompt = prompt
    def mk_input(self, query, related_chunks, k):
        context_str = "\n\n".join(related_chunks[:k])
        user_content = self.prompt.format(context_str=context_str, query=query)
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                   ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return inputs
    @torch.inference_mode()
    def invoke(self, query):
        related_chunks = self.database.retrieve(query)
        inputs = self.mk_input(query, related_chunks, self.database.k)
        config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=512
        )
        generated_ids = self.model.generate(**inputs, generation_config=config)
        # 生成されたID（output_ids）から入力部分の長さを削る
        generated_ids = generated_ids[0][len(inputs.input_ids[0]):]
        # デコードして文字列にする
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
```

・pp.25-27 ファイル名 create_db.py の "_" が抜けています (2026年7月23日)

・pp.28-29 ファイル名 generate_ans.py の "_" が抜けています (2026年7月23日)

・p.55 ファイル名が mm_rag_image_to_text.py となっているが、mk_chunks.py の間違い (2026年7月23日)

・p.105 ファイル名が mk-rag-db-from-text.py となっているが、sample_langgraph.py の間違い (2026年7月23日)

・p.116 ファイル名が mk-rag-db-from-text.py となっているが、mm_rag.py の間違い (2026年7月23日)

・p.118

（誤）(1) 4章で利用した data ディレクトリをまるごとコピーする（リンクを張ってもよい）

（正）(1) 4章で利用した data ディレクトリと doc ディレクトリをまるごとコピーする（リンクを張ってもよい）

・p.116 の2つのコードを含むファイルが mk-rag-db-from-text.py になっているが、mma_rag.py の間違い。(2026年6月26日)
