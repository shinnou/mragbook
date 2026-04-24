# -*- coding: sjis -*-

import json
import openai

with open("data/ibaraki_vec_db.json", "r", encoding="utf-8") as f:
    config = json.load(f)

vector_store_id = config.get("vector_store_id")

query = "日立駅を設計した人は誰ですか？"

client = openai.OpenAI()

# instructions = """
# あなたは検索結果（file_search）で提供される文書の内容のみに基づいて回答するアシスタントです。

# - 検索結果に明確な根拠がある場合のみ回答してください。
# - 検索結果に答えが含まれていない場合は、「分かりません」と答えてください。
# - 一般知識や推測で補完してはいけません。
# - 回答は簡潔かつ正確に述べてください。
#"""

instructions = """
あなたは提供された検索結果（file_search）のみを使用して質問に答えるAIアシスタントです。
以下のルールを厳守してください：
- 検索結果に含まれていない情報は絶対に回答に含めないでください。
- 自分の知識や一般常識を使用しないでください。
- 検索結果から回答が見つからない場合は、正直に「提供された情報からは回答できません」と答えてください。
- 回答は簡潔かつ具体的に記述してください。
"""

response = client.responses.create(
#    model="gpt-4.1-mini",
    model="gpt-4.1",
    instructions=instructions,
    input=query,
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        }
    ],
)

print("\n--- 回答 ---")
print(response.output_text)
