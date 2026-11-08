"""第3回 宿題(基礎): INSTRUCTIONS / docstring を1か所ずつ変えて、同じ質問のふるまいを比べる。

app/ のファイルは書き換えず、ここで変更版のエージェントを組み立てる（変更は1実験につき1か所）。
実行には GEMINI_API_KEY が必要。結果は lectures/03/basic_results.md に追記される。

    uv run python scripts/lec03_basic_experiment.py            # 全実験
    uv run python scripts/lec03_basic_experiment.py E1 E2      # 指定した実験だけ
"""

import inspect
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # python scripts/… で直接動かしても app を読めるように

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.messages import ToolCallPart  # noqa: E402

from app import tools  # noqa: E402
from app.agent import INSTRUCTIONS, build_model  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "lectures" / "03" / "basic_results.md"
Q_COMPARE = "LoRAとQLoRAの違いは？"
Q_LEVELS = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
PAUSE = 12  # 試行の間の待ち秒数（無料枠は1分15リクエスト。1試行で2〜3リクエスト使う）
NL = chr(10)
REPEAT = 3  # LLM の出力は毎回揺れるので、同じ条件を複数回走らせる

SEARCH_LINE = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
LAST_LINE = "- 回答の最後に、関係する回（第N回）を示す。\n"
ONE_TERM = 'query には調べたい用語を1つだけ入れる（例: "LoRA"）'


def build(instructions=INSTRUCTIONS, search_doc=None) -> Agent:
    a = Agent(instructions=instructions)
    doc = inspect.cleandoc(tools.search_glossary.__doc__)
    a.tool_plain(description=search_doc or doc)(tools.search_glossary)
    a.tool_plain(tools.get_lecture_summary)
    return a


def changed(old: str, new: str, text: str) -> str:
    assert old in text, f"変更箇所が見つかりません: {old!r}"
    return text.replace(old, new)


SEARCH_DOC = inspect.cleandoc(tools.search_glossary.__doc__)
EXPERIMENTS = {
    "E0": ("変更なし（基準）", build(), [Q_COMPARE, Q_LEVELS]),
    "E1": ("INSTRUCTIONS から「必ず search_glossary で確認する」行を削除",
           build(instructions=changed(SEARCH_LINE, "", INSTRUCTIONS)), [Q_COMPARE, Q_LEVELS]),
    # E1 で検索が止まらなかった理由の切り分け（INSTRUCTIONS と docstring のどちらが効いているか）
    "E4": ("INSTRUCTIONS の検索の行 と docstring の「答える前に使うこと」を両方削除",
           build(instructions=changed(SEARCH_LINE, "", INSTRUCTIONS),
                 search_doc=changed("について\n答える前に使うこと。", "について。", SEARCH_DOC)), [Q_COMPARE, Q_LEVELS]),
    "E5": ("docstring の「答える前に使うこと」だけ削除（INSTRUCTIONS は元のまま）",
           build(search_doc=changed("について\n答える前に使うこと。", "について。", SEARCH_DOC)), [Q_COMPARE, Q_LEVELS]),
    "E2": ("search_glossary の docstring「1つだけ」→「まとめて入れる」",
           build(search_doc=changed(ONE_TERM, 'query には調べたい用語をまとめて入れる（例: "LoRA QLoRA"）', SEARCH_DOC)),
           [Q_COMPARE]),
    "E3": ("INSTRUCTIONS の「最後に第N回を示す」→「最後に確認問題を1問出す」",
           build(instructions=changed(LAST_LINE, "- 回答の最後に、確認問題を1問出す。\n", INSTRUCTIONS)), [Q_COMPARE]),
}


def run_with_retry(target, q, model):
    """無料枠(1分15リクエスト)の 429 と、混雑時の 503 は待って再試行する。"""
    for attempt in range(6):
        try:
            return target.run_sync(q, model=model)
        except ModelHTTPError as e:
            if e.status_code not in (429, 503) or attempt == 5:
                raise
            print("  429/503: 60秒待って再試行します")
            time.sleep(60)


def main() -> None:
    names = sys.argv[1:] or list(EXPERIMENTS)
    model = build_model()
    with OUT.open("a", encoding="utf-8") as f:  # 途中で止まっても残るよう、1試行ごとに書き出す
        f.write(f"{NL}## 実行 {datetime.now():%Y-%m-%d %H:%M}（model={model.model_name}）{NL}{NL}")
        for name in names:
            title, target, questions = EXPERIMENTS[name]
            for q in questions:
                for i in range(1, REPEAT + 1):
                    result = run_with_retry(target, q, model)
                    calls = [f"{p.tool_name}({p.args})" for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
                    quoted = "> " + str(result.output).replace(NL, NL + "> ")
                    f.write(NL.join([f"### {name} {title} / 試行{i}", f"- 質問: {q}", f"- requests: {result.usage.requests}",
                                     f"- tool call: {calls or 'なし'}", "- 回答:", "", quoted, "", ""]))
                    f.flush()
                    print(f"{name} #{i} {q[:12]} requests={result.usage.requests} calls={len(calls)}")
                    time.sleep(PAUSE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
