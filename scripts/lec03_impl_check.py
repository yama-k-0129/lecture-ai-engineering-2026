"""第3回 宿題(実装): save_learning_log を実際の LLM で動かし、承認・拒否の両方を確かめる。

承認の質問(input)には、このスクリプトが y / n を自動で答える（実際の CLI は app.agent の ask_approvals）。
実行には GEMINI_API_KEY が必要。結果は lectures/03/impl_results.md に書き出される。
保存されるファイルは learning_log/（.gitignore 済み）。
"""

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.messages import ToolCallPart, ToolReturnPart  # noqa: E402

from app import tools  # noqa: E402
from app.agent import agent, build_model, run_with_approval  # noqa: E402

OUT = ROOT / "lectures" / "03" / "impl_results.md"
NL = chr(10)
QUESTION = "今日の学習を記録して。内容は「LoRAとQLoRAの違いを学んだ。量子化でメモリが減る」です。"


def run(approve: bool):
    log = []

    def ask(requests: DeferredToolRequests) -> DeferredToolResults:
        results = DeferredToolResults()
        for call in requests.approvals:
            log.append(f"承認待ち: {call.tool_name}({call.args}) -> {'y' if approve else 'n'}")
            results.approvals[call.tool_call_id] = True if approve else ToolDenied("利用者が実行を拒否しました")
        return results

    for attempt in range(6):
        try:
            return log, run_with_approval(agent, QUESTION, build_model(), ask=ask)
        except ModelHTTPError as e:
            if e.status_code != 429 or attempt == 5:
                raise
            print("  429: 60秒待って再試行します")
            time.sleep(60)


def main() -> None:
    with OUT.open("a", encoding="utf-8") as f:
        f.write(f"{NL}## 実行 {datetime.now():%Y-%m-%d %H:%M}{NL}{NL}")
        for approve in (True, False):
            before = set(tools.LEARNING_LOG_DIR.glob("*.md")) if tools.LEARNING_LOG_DIR.exists() else set()
            log, result = run(approve)
            after = set(tools.LEARNING_LOG_DIR.glob("*.md")) if tools.LEARNING_LOG_DIR.exists() else set()
            new = sorted(p.name for p in after - before)
            msgs = [p for m in result.all_messages() for p in m.parts]
            calls = [f"{p.tool_name}({p.args})" for p in msgs if isinstance(p, ToolCallPart)]
            returns = [f"{p.tool_name} -> {str(p.content)[:100]}" for p in msgs if isinstance(p, ToolReturnPart)]
            quoted = "> " + str(result.output).replace(NL, NL + "> ")
            f.write(NL.join([f"### {'承認(y)' if approve else '拒否(n)'}", f"- 承認の質問: {log or 'なし'}",
                             f"- tool call: {calls}", f"- tool return: {returns}", f"- requests: {result.usage.requests}",
                             f"- 新しくできたファイル: {new or 'なし'}", "- 最終回答:", "", quoted, "", ""]))
            f.flush()
            print(f"{'approve' if approve else 'deny'}: new files={new} calls={len(calls)}")
            time.sleep(15)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
