from types import SimpleNamespace

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "入力トークン=12" in output
    assert "出力トークン=34" in output
    assert "応答時間=" in output


def test_chat_reuses_conversation_for_multiple_questions(monkeypatch, capsys):
    sent_messages = []
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
        ),
        text="回答",
    )
    conversation = SimpleNamespace(
        send_message=lambda question: sent_messages.append(question) or response
    )
    client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: conversation)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    questions = iter(["最初の質問", "続きの質問", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask.chat()

    assert sent_messages == ["最初の質問", "続きの質問"]
    assert capsys.readouterr().out.count("応答時間=") == 2


def test_compare_runs_each_question_with_each_model(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        ask,
        "ask",
        lambda question, model: calls.append((question, model)) or "回答",
    )

    ask.compare(["model-a", "model-b"], ["質問1", "質問2"])

    assert calls == [
        ("質問1", "model-a"),
        ("質問1", "model-b"),
        ("質問2", "model-a"),
        ("質問2", "model-b"),
    ]
    assert capsys.readouterr().out.count("--- model=") == 4


def test_compare_continues_after_api_error(monkeypatch, capsys):
    calls = []

    def ask_with_error(question, model):
        calls.append((question, model))
        if model == "model-a":
            raise ask.errors.APIError(code=503, response_json={"message": "busy"})
        return "回答"

    monkeypatch.setattr(ask, "ask", ask_with_error)

    ask.compare(["model-a", "model-b"], ["質問"])

    assert calls == [("質問", "model-a"), ("質問", "model-b")]
    output = capsys.readouterr().out
    assert "混み合っています" in output
    assert "回答" in output