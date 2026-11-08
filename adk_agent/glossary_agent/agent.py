"""第3回 発展(a): 演習のエージェントを Google ADK で組み立て直したもの。

INSTRUCTIONS とツール（app/tools.py）は Pydantic AI 版とまったく同じものを使う。
違うのは「エージェントを組み立てる部分」だけ。

実行（リポジトリ直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS

root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    # 承認付きの save_learning_log は対象外（課題の範囲外）。普通の関数をそのまま渡せる
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
