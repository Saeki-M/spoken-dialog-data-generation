import argparse
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ===== 設定: OpenAI(=Ollama) クライアント =====
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API
    api_key="ollama",  # ダミーキー
)

# ===== 20種類の「よくある対話タイプ」 =====
TASK_TYPES = [
    "レストラン予約",
    "天気案内（現在/今日/週）",
    "時刻表検索（電車/バス）",
    "乗換案内（最短/安い/本数）",
    "ホテル予約（条件・日付・人数）",
    "フライト情報（便名/遅延/到着）",
    "タクシー配車/配車アプリ連携",
    "商品検索・価格比較（買い物）",
    "返品/返金手続きサポート",
    "カレンダー予定の作成/確認",
    "リマインダー設定/管理",
    "ニュース要約と深掘りQA",
    "一般知識QA（百科/豆知識）",
    "翻訳（短文/用途指定）",
    "道案内・住所/地図検索",
    "サブスク解約/プラン変更支援",
    "技術サポート（簡易トラブルシュート）",
    "健康/病院予約（科/時間/場所）",
    "家事タスク計画（買い物リスト等）",
    "旅行計画（行程/見どころ/予算）",
]


# ===== プロンプト生成 =====
def build_prompt(task_type: str, target_turns: int = 20) -> str:
    return f"""
あなたは頼りになるAIアシスタントです。
ユーザと音声で話しているような自然な日本語会話を作ってください。

# 会話ゴール
- 対話タイプ: 「{task_type}」
- 目的: ユーザがこのタスクを完了または理解できるように支援する
- 会話は最大で{target_turns}発話程度（少なくてもよい）

# 会話スタイル
- 話し言葉で自然に。文は短めでテンポ良く。
- 「。」「、」「？」「！」などの句読点は使ってよい。
- 難しい文語表現は避ける。

# 出力仕様（最重要）
- **必ず JSON 配列のみ**を出力（説明やコードブロックは出力しない）。
- JSON 構文（[ ] {{ }} , : "）は使用する。ただし **content の中では ASCII のダブルクォート " と バックスラッシュ \\ と 改行 を使わない。**
  - 引用が必要なら日本語の鉤括弧「」を使うこと。
  - content は1行で書くこと（\\n も不可）。
- 形式例：
[
  {{"role":"assistant","content":"こんにちは。今日は何について話しますか？"}},
  {{"role":"user","content":"（ユーザが{task_type}の意図を述べる）"}}
]

# 会話の流れ
- 最初はアシスタントが「こんにちは。今日は何について話しますか？」。
- ユーザは最初の発話で {task_type} に関する要件を述べる。
- アシスタントは確認・提案・要約を挟みつつタスクを進める。
- 終盤で結果を確認し、自然に終了する（無理に {target_turns} に合わせない）。
""".strip()


# ===== モデル呼び出し =====
def generate_dialog(
    task_type: str, model: str = "gpt-oss:20b", turns: int = 20
) -> list[dict]:
    prompt = build_prompt(task_type, turns)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Japanese-speaking assistant that generates high-quality simulated dialogues (JSON array only).",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=4096,
    )

    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re

        content_clean = re.sub(
            r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE
        )
        data = json.loads(content_clean)

    fixed = []
    for m in data:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = m.get("content")
        if role in ("assistant", "user") and isinstance(text, str) and text.strip():
            fixed.append({"role": role, "content": text.strip()})

    # 👇 ここを変更：
    # 不足分を強制的に埋めない。生成された分だけ使う。
    # 必要なら turns 上限で切り捨て。
    if len(fixed) > turns:
        fixed = fixed[:turns]

    return fixed


# ===== TSV保存 =====
def save_dialog_as_tsv(
    dialog: list[dict], out_path: Path, task_type: str, meta: dict | None = None
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        # ヘッダ
        headers = ["turn_index", "role", "content", "task_type"]
        if meta:
            # メタ情報を列として追加（例: seed, timestamp, model）
            headers.extend(meta.keys())
        writer.writerow(headers)

        # 本文
        for i, msg in enumerate(dialog, start=1):
            row = [i, msg["role"], msg["content"], task_type]
            if meta:
                row.extend(meta.values())
            writer.writerow(row)


# ===== メイン =====
def main():
    parser = argparse.ArgumentParser(
        description="Generate n simulated dialogues and save each as TSV."
    )
    parser.add_argument("--n", type=int, default=1, help="生成する対話数（初期値: 1）")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="モデル名（初期値: gpt-oss:20b）",
    )
    parser.add_argument(
        "--turns", type=int, default=20, help="1対話あたりの発話数（初期値: 20）"
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs_tsv", help="出力ディレクトリ"
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(1, args.n + 1):
        # ランダムにタスクタイプを選択
        task_type = random.choice(TASK_TYPES)

        # 生成
        dialog = generate_dialog(
            task_type=task_type, model=args.model, turns=args.turns
        )

        # メタ情報
        meta = {
            "seed_like": str(random.randint(0, 2**31 - 1)),
            "timestamp": now,
            "model": args.model,
        }

        # ファイル名: 連番 + タスクタイプスラッグ
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", task_type).strip("_")
        fname = f"dialog_{i:04d}_{slug}.tsv"
        save_dialog_as_tsv(dialog, out_dir / fname, task_type, meta=meta)

        print(f"✅ Saved: {out_dir / fname}  (task_type={task_type})")


if __name__ == "__main__":
    main()
