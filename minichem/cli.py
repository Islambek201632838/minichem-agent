"""Command line: python -m minichem.cli "your question" """

from __future__ import annotations

import argparse
import asyncio

from minichem.assistant import open_assistant


async def _run(question: str, session_id: str, verbose: bool) -> None:
    async with open_assistant() as assistant:
        result = await assistant.ask(question, session_id)
    if verbose:
        for call in result.tool_calls:
            status = f"ERROR: {call.error}" if call.error else "ok"
            print(f"[{call.agent}] {call.tool}({call.arguments}) -> {status}")
        print()
    print(result.answer)
    if verbose:
        print(
            f"\n{result.latency_s}s, {result.llm_calls} LLM calls, "
            f"{result.input_tokens}+{result.output_tokens} tokens"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="minichem", description="Ask the chemistry assistant.")
    parser.add_argument("question")
    parser.add_argument("--session", default="cli", help="session id (memory key)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool calls and usage")
    args = parser.parse_args()
    asyncio.run(_run(args.question, args.session, args.verbose))


if __name__ == "__main__":
    main()
