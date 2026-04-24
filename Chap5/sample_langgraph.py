import random
from operator import add
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# ===== 1) State 定義 =====
class PrimeState(TypedDict):
    attempts: int                 # 乱数生成回数
    last_n: int                   # 直近に引いた乱数
    primes: Annotated[list[int], add]  # 素数リスト（"追加"でマージさせる）
    logs: Annotated[list[str], add]    # ログも同様に追加マージ

MAX_ATTEMPTS = 10
TARGET_PRIMES = 2

# ===== 2) 素数判定（ふつうの関数でOK）=====
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

# ===== 3) ノード関数 =====
def draw(state: PrimeState) -> PrimeState:
    n = random.randint(1, 1000)
    return {
        "attempts": state["attempts"] + 1,
        "last_n": n,
        "logs": [f"[draw] attempts={state['attempts'] + 1}, n={n}"],
    }


def judge(state: PrimeState) -> PrimeState:
    n = state["last_n"]
    if is_prime(n):
        return {
            "primes": [n],  # Annotated[..., add] なので「追加」される
            "logs": [f"[judge] {n} is PRIME -> appended"],
        }
    return {
        "logs": [f"[judge] {n} is not prime"],
    }


# ===== 4) 条件分岐（次にどこへ行くか）=====
def route(state: PrimeState) -> Literal["continue", "end"]:
    if len(state["primes"]) >= TARGET_PRIMES:
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"
    return "continue"


# ===== 5) グラフ構築 =====
builder = StateGraph(PrimeState)

builder.add_node("draw", draw)
builder.add_node("judge", judge)

builder.add_edge(START, "draw")
builder.add_edge("draw", "judge")

builder.add_conditional_edges(
    "judge",
    route,
    {
        "continue": "draw",
        "end": END,
    },
)

graph = builder.compile()

### グラフを視覚化して画像に保存
#
# png_bytes = graph.get_graph().draw_mermaid_png()
#
# out_path = "prime_stategraph.png"
# with open(out_path, "wb") as f:
#     f.write(png_bytes)
#
# print(f"saved: {out_path}")
#

### プログラムの実行

init_state: PrimeState = {
    "attempts": 0,
    "last_n": 0,
    "primes": [],
    "logs": [],
}

out = graph.invoke(init_state)

print("=== logs ===")
for line in out["logs"]:
    print(line)

print("\n=== result ===")
print(f"attempts: {out['attempts']} / {MAX_ATTEMPTS}")
print(f"primes_found ({len(out['primes'])}): {out['primes']}")
