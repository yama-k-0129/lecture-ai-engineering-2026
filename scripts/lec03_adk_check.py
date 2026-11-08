"""第3回 発展(a): ADK 版エージェントに演習と同じ質問を投げ、ツール呼び出しと回答を記録する。

`adk run` は対話式なので、同じことを Runner で非対話に実行する（エージェント本体は同じ root_agent）。
実行には GEMINI_API_KEY と google-adk が必要:  pip install google-adk
結果は lectures/03/adk_results.md に1質問ごとに書き出される。
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))  # ADK が読む名前

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from adk_agent.glossary_agent.agent import root_agent  # noqa: E402

OUT = ROOT / "lectures" / "03" / "adk_results.md"
NL = chr(10)
QUESTIONS = [
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "LoRAとQLoRAの違いは？",
    "第2回の内容から確認問題を1問出して",
    "今日の学習を記録して。内容は「LoRAを学んだ」です。",  # ADK 版には save_learning_log を渡していない
]
PAUSE = 12


async def ask(question: str):
    runner = InMemoryRunner(agent=root_agent, app_name="lec03")
    await runner.session_service.create_session(app_name="lec03", user_id="u", session_id="s")
    calls, answer, llm_events = [], "", 0
    message = types.Content(role="user", parts=[types.Part(text=question)])
    async for event in runner.run_async(user_id="u", session_id="s", new_message=message):
        for c in event.get_function_calls():
            calls.append(f"{c.name}({dict(c.args or {})})")
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(p.text or "" for p in event.content.parts)
    return calls, answer


def main() -> None:
    with OUT.open("a", encoding="utf-8") as f:
        f.write(f"{NL}## 実行 {datetime.now():%Y-%m-%d %H:%M}（ADK / model={root_agent.model}）{NL}{NL}")
        for q in QUESTIONS:
            for attempt in range(6):
                try:
                    calls, answer = asyncio.run(ask(q))
                    break
                except Exception as e:  # 429 は待って再試行（無料枠は1分15リクエスト）
                    if "429" not in str(e) or attempt == 5:
                        raise
                    print("  429: 60秒待って再試行します")
                    time.sleep(60)
            quoted = "> " + answer.replace(NL, NL + "> ")
            f.write(NL.join([f"### {q}", f"- tool call: {calls or 'なし'}", "- 回答:", "", quoted, "", ""]))
            f.flush()
            print(f"{q[:16]} calls={len(calls)}")
            time.sleep(PAUSE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
