"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_metrics(model: str, elapsed: float, usage: object | None) -> None:
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    print(
        f"[計測] model={model} 応答時間={elapsed:.2f}秒 "
        f"入力トークン={input_tokens if input_tokens is not None else '不明'} "
        f"出力トークン={output_tokens if output_tokens is not None else '不明'}"
    )


def ask(question: str, model: str = MODEL) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    _print_metrics(model, perf_counter() - started_at, response.usage_metadata)

    return response.text


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)
    print(f"[model] {MODEL} (終了: exit / quit / Ctrl+D)")

    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        started_at = perf_counter()
        response = conversation.send_message(question)
        _print_metrics(MODEL, perf_counter() - started_at, response.usage_metadata)
        print(response.text)


def compare(models: list[str], questions: list[str]) -> None:
    for question in questions:
        for model in models:
            print(f"\n--- model={model} / 質問={question} ---")
            try:
                print(ask(question, model=model))
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    modes.add_argument("--compare", action="store_true", help="複数モデル・質問を逐次比較する")
    parser.add_argument("--models", nargs="+", help="比較するモデル名（--compare時に2つ以上）")
    parser.add_argument("--questions", nargs="+", help="比較する質問（--compare時、省略時は既定の質問）")
    parser.add_argument("question", nargs="?", help="質問文")
    args = parser.parse_args()

    try:
        if args.compare:
            if not args.models or len(args.models) < 2:
                parser.error("--compare には --models で2つ以上のモデルを指定してください")
            compare(args.models, args.questions or [args.question or DEFAULT_QUESTION])
        elif args.chat:
            chat()
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[model] {MODEL}")
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
