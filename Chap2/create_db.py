import openai
client = openai.OpenAI()

# ベクトルストアを作成
vector_store = client.vector_stores.create(
    name="茨城関連の記事",
    expires_after={"anchor": "last_active_at", "days": 7},  # 7日間使わなければ自動削除
)

# フィイルをアップロード & チャンク化・ベクトル化
# upload_and_poll は処理が完了するまで待つ

with open("data/ibaraki.txt", "rb") as f:
    client.vector_stores.files.upload_and_poll(
        vector_store_id=vector_store.id,
        file=f
    )

# rag_config.json に vector_store_id を保存
config = {
    "vector_store_id": vector_store.id,
    "vector_store_name": vector_store.name,
}

import json

with open("data/ibaraki_vec_db.json", "w", encoding="utf-8") as f:
    json.dump(config, f)
