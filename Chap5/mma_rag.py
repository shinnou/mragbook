#---------------------------------------------------
#
# ColPali
#
#---------------------------------------------------

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

class MyColPali:
    def __init__(self, model, processor, embedding):
        self.model = model
        self.processor = processor
        self.embedding = embedding
    def search(self, query, k):
        p_query = self.processor.process_queries([ query ])
        p_query.to(self.model.device)
        with torch.inference_mode():
            query_embedding = model(**p_query)
        query_embedding = query_embedding.to(dtype=torch.bfloat16)
        scores = self.processor.score_multi_vector(query_embedding, self.embedding)[0]
        values, indices = torch.topk(scores, k)
        result = []
        for i in range(k):
            ele = [0, 0.0, ""]
            ele[0] = indices[i] + 1
            ele[1] = values[i]
            ele[2] = f"doc/page_{indices[i] + 1}.png"
            result.append(ele)
        return result

repo_id = "vidore/colqwen2.5-v0.2"

model = ColQwen2_5.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="auto" ,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    repo_id,
    use_fast=True
)

emb_path ="data/embedding.pt"
embedding = torch.load(emb_path)

colpali_search = MyColPali(model, processor, embedding)

#---------------------------------------------------
#
# VLM  (Qwen2.5-VL-7B-Instruct)
#
#---------------------------------------------------

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import requests

vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_model_id, dtype="auto", device_map="auto"
).eval()

# vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     vlm_model_id, dtype=torch.bfloat16, device_map="auto"
# ).eval()

processor = AutoProcessor.from_pretrained(vlm_model_id, use_fast=True)

system_prompt = """
あなたは技術文書や統計的な調査報告書を理解を支援する専門アシスタントです。
"""
user_prompt_template = """
添付の画像は、ある PDF 文書の１ページです。そのページにある画像も含めて、このページの内容を理解し、ユーザからの質問に、できるだけ丁寧に、かつ正確に、回答して下さい。

### ユーザからの質問
{query}
"""

def vlm_qa(query, image_path):
    user_prompt = user_prompt_template.format(query=query)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
             "role": "user",
             "content": [
                  {"type": "image", "image": image_path},
                  {"type": "text", "text": user_prompt}
             ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = len(inputs["input_ids"][0])
    with torch.inference_mode():
        generation = vlm_model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    ans_ids = generation[0][input_len:]
    answer = processor.decode(ans_ids, skip_special_tokens=True)
    return answer

#---------------------------------------------------
#
# LLM 判定、
#
#---------------------------------------------------

import json
import operator
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# -----------------------------
#  関数
# -----------------------------

def search_pages(query: str, k: int) -> List[Tuple[Any, float, str, Any]]:
    return colpali_search.search(query, k)

# -----------------------------
# judge (別LLM) のためのスキーマ
# -----------------------------

class JudgeOutput(BaseModel):
    is_good: bool = Field(..., description="回答が質問に対して妥当なら true")
    confidence: float = Field(..., ge=0.0, le=1.0, description="判定の自信（0?1）")
    reasons: List[str] = Field(default_factory=list, description="判定理由（短い箇条書き）")
    missing_points: List[str] = Field(default_factory=list, description="不足・曖昧な点")
    rewrite_query_hint: Optional[str] = Field(None, description="次の候補ページで試す際の改善ヒント（任意）")


import ollama

def judge_answer(question: str, answer: str, page_id: Any, score: float) -> Dict[str, Any]:
    """
    別LLMで「回答が適切か」を判定する。
    戻り値は dict で state に格納しやすくする（JudgeOutput互換）。
    """
    # debug  のため以下を出力
    print("-- 質問 ", "---")
    print(question)
    print("-- M-RAG の回答 ", "---")
    print(answer)
    # -----------------------
    system = (
        "あなたは厳格な回答評価者です。与えられた回答が質問に十分に答えているかを判定してください。"
        "不足がある場合は is_good=false にします。推測や根拠不明の断定は減点します。"
        "出力は必ず JSON のみで返してください。"
    )

    user = f"""
[質問]
{question}

[候補ページ情報]
page_id: {page_id}
retrieval_score: {score}

[VLMの回答]
{answer}

判定基準:
- 質問の要求(数値/比較/定義/範囲/条件)を満たしているか
- 「何をもってそう言えるか」が明確か（根拠不明の断定はNG）
- 重要な抜けがないか
- 文章が極端に短い/一般論だけ/「わかりません」系はNG

次のJSON形式で返して:
{{
  "is_good": true/false,
  "confidence": 0.0-1.0,
  "reasons": ["..."],
  "missing_points": ["..."],
  "rewrite_query_hint": "..."  // 任意
}}
"""

    client = ollama.Client(host='http://157.80.86.245:11434')
    response = client.chat(model='qwen2.5:7b', messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": user},
    ])

    text = response['message']['content']
    # debug  のため以下を出力
    print("-- JUDGE の判定 ", "---")
    print(text)
    #------------------------
    data = json.loads(text)
    validated = JudgeOutput(**data)
    return validated.model_dump()

# -----------------------------
# LangGraph 状態定義
# -----------------------------

class RetryState(TypedDict, total=False):
    question: str
    k: int                      # 最大何件まで試すか（通常 3）
    hits: List[Dict[str, Any]]  # 検索結果を dict 化して保存
    attempt: int                # 0,1,2... 何番目の hit を試しているか
    current_hit: Dict[str, Any]
    answer: str
    judge: Dict[str, Any]
    history: Annotated[List[Dict[str, Any]], operator.add]
    final_answer: str

# -----------------------------
# ノード実装
# -----------------------------

def node_retrieve(state: RetryState) -> Dict[str, Any]:
    question = state["question"]
    k = int(state.get("k", 3))
    raw_hits = search_pages(question, k)
    hits: List[Dict[str, Any]] = []
    for h in raw_hits:
        # 期待: (page_id, score, image_path, ...)
        page_id = h[0]
        score = float(h[1])
        image_path = h[2]
        hits.append(
            {
                "page_id": page_id,
                "score": score,
                "image_path": image_path,
                "raw": h,
            }
        )
    return {"hits": hits, "attempt": int(state.get("attempt", 0))}

def node_pick_hit(state: RetryState) -> Dict[str, Any]:
    attempt = int(state.get("attempt", 0))
    hits = state.get("hits", [])
    if attempt >= len(hits):
        return {"current_hit": {"page_id": None, "score": 0.0, "image_path": None}}
    return {"current_hit": hits[attempt]}

def node_generate_vlm(state: RetryState) -> Dict[str, Any]:
    question = state["question"]
    hit = state["current_hit"]
    image_path = hit.get("image_path")
    ans = vlm_qa(question, image_path)
    return {"answer": ans}

def node_judge(state: RetryState) -> Dict[str, Any]:
    question = state["question"]
    ans = state.get("answer", "")
    hit = state.get("current_hit", {})
    page_id = hit.get("page_id")
    score = float(hit.get("score", 0.0))
    judge = judge_answer(question, ans, page_id=page_id, score=score)
    record = {
        "attempt": int(state.get("attempt", 0)),
        "page_id": page_id,
        "score": score,
        "image_path": hit.get("image_path"),
        "answer": ans,
        "judge": judge,
    }
    return {"judge": judge, "history": [record]}

def node_bump_attempt(state: RetryState) -> Dict[str, Any]:
    return {"attempt": int(state.get("attempt", 0)) + 1}

def node_finalize(state: RetryState) -> Dict[str, Any]:
    """
    accept のときも exhausted のときもここに来る。
    exhausted の場合は judge を添えて「低信頼」表示にしてもよい。
    """
    ans = state.get("answer", "")
    hit = state.get("current_hit", {})
    judge = state.get("judge", {})
    page_id = hit.get("page_id")
    score = hit.get("score")
    # お好みで provenance や判定理由を付与
    suffix = ""
    if judge and not judge.get("is_good", False):
        suffix = "\n\n（注）判定LLMは「回答が不十分の可能性あり」と評価しました。"
    final = f"{ans}\n\n---\n参照ページ: {page_id}（score={score}）{suffix}"
    return {"final_answer": final}

# -----------------------------
# 条件分岐（retry / accept / exhausted）
# -----------------------------
def route_after_judge(state: RetryState) -> str:
    judge = state.get("judge", {})
    is_good = bool(judge.get("is_good", False))
    attempt = int(state.get("attempt", 0))
    k = int(state.get("k", 3))
    hits = state.get("hits", [])
    # accept
    if is_good:
        return "accept"
    # exhausted: もう試せない
    # - attempt は現在試した index
    # - 次へ進むには attempt+1 < min(k, len(hits)) が必要
    if (attempt + 1) >= min(k, len(hits)):
        return "exhausted"
    # 上記以外は retry
    return "retry"

# -----------------------------
# グラフ組み立て
# -----------------------------
def build_graph():
    #---------------------------
    # State の定義
    #---------------------------
    g = StateGraph(RetryState)
    #---------------------------
    # ノードの設定
    #---------------------------
    g.add_node("retrieve", node_retrieve)
    g.add_node("pick_hit", node_pick_hit)
    g.add_node("generate_vlm", node_generate_vlm)
    g.add_node("judge", node_judge)
    g.add_node("bump_attempt", node_bump_attempt)
    g.add_node("finalize", node_finalize)
    #---------------------------
    # エッジの設定-1
    #---------------------------
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "pick_hit")
    g.add_edge("pick_hit", "generate_vlm")
    g.add_edge("generate_vlm", "judge")
    #---------------------------
    # 条件分岐
    #---------------------------
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "accept": "finalize",
            "retry": "bump_attempt",
            "exhausted": "finalize",
        },
    )
    #---------------------------
    # エッジの設定-2
    #---------------------------
    g.add_edge("bump_attempt", "pick_hit")
    g.add_edge("finalize", END)
    #---------------------------
    # グラフ構築
    #---------------------------
    return g.compile()

# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    app = build_graph()
    question = "Twitterをニュース目的で利用している割合が一番大きいのはどの年代か"  ## p.105
    result = app.invoke(
        {
            "question": question,
            "k": 3,
            "attempt": 0,
            "history": [],  # reducer を使うので初期化推奨
        }
    )
    print("====== FINAL ANSWER ======")
    print(result["final_answer"])
    print("\n====== HISTORY (debug) ======")
    for h in result.get("history", []):
        print(f"""
- attempt={h['attempt']},
  page_id={h['page_id']},
  score={h['score']},
  is_good={h['judge'].get('is_good')}
        """)
