"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval_and_runs_after_yes(tmp_path, monkeypatch):
    from pydantic_ai import DeferredToolResults

    from app import tools
    from app.agent import run_with_approval

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    asked = []

    def say_yes(requests):
        asked.extend(c.tool_name for c in requests.approvals)
        return DeferredToolResults(approvals={c.tool_call_id: True for c in requests.approvals})

    with agent.override(model=TestModel(call_tools=["save_learning_log"])):
        result = run_with_approval(agent, "今日の学習を記録して", model=None, ask=say_yes)
    assert asked == ["save_learning_log"]
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert not isinstance(result.output, DeferredToolRequests)


def test_save_learning_log_is_not_run_when_denied(tmp_path, monkeypatch):
    from pydantic_ai import DeferredToolResults, ToolDenied

    from app import tools
    from app.agent import run_with_approval

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)

    def say_no(requests):
        return DeferredToolResults(approvals={c.tool_call_id: ToolDenied("no") for c in requests.approvals})

    with agent.override(model=TestModel(call_tools=["save_learning_log"])):
        run_with_approval(agent, "記録して", model=None, ask=say_no)
    assert list(tmp_path.glob("*.md")) == []
