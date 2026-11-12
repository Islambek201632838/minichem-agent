"""Run the golden suite against the real model and report success, errors, latency and tokens.

uv run python -m eval.run_eval                  # 3 runs per question, live PubChem
uv run python -m eval.run_eval --mock-tools     # offline PubChem fixtures, isolates the agents
uv run python -m eval.run_eval --ids t01,w03 --runs 1
uv run python -m eval.run_eval --file eval/generated.yaml --runs 1   # generated questions
uv run python -m eval.run_eval --file eval/complex.yaml --runs 1     # drug-sized molecules
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from eval.checks import classify, match_tools, run_checks
from minichem.assistant import ChatResult, ChemAssistant, open_assistant
from minichem.config import PROJECT_ROOT, Settings, get_settings

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTCOMES = ("complete", "partial", "incorrect")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Golden entries with `question` normalised to a list of turns."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in entries:
        entry["turns"] = entry.get("turns") or [entry["question"]]
    return entries


async def run_once(
    assistant: ChemAssistant, entry: dict[str, Any], run: int, tag: str
) -> dict[str, Any]:
    """One run of one golden entry. Only the last turn is judged; earlier turns set up memory."""
    session = f"{entry['id']}-r{run}-{tag}"
    record: dict[str, Any] = {"id": entry["id"], "run": run, "error": None}
    result: ChatResult | None = None
    latency = 0.0
    tokens = [0, 0]
    try:
        for turn in entry["turns"]:
            result = await assistant.ask(turn, session)
            latency += result.latency_s
            tokens = [tokens[0] + result.input_tokens, tokens[1] + result.output_tokens]
    except Exception as exc:  # noqa: BLE001 - one failed run must not abort the whole evaluation
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    tools = [c.tool for c in result.tool_calls] if result else []
    tool_match = match_tools(tools, entry["expected_tools"])
    failures = run_checks(result.answer, entry.get("checks")) if result else None
    record.update(
        outcome=classify(tool_match, failures, crashed=record["error"] is not None),
        tool_match=tool_match,
        answer_failures=failures,
        tools=tools,
        tool_errors=sum(1 for c in result.tool_calls if c.error) if result else 0,
        warnings=result.warnings if result else [],
        latency_s=round(latency, 2),
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        answer=result.answer if result else "",
    )
    return record


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of run records."""
    n = len(records)
    counts = {o: sum(r["outcome"] == o for r in records) for o in OUTCOMES}
    return {
        "runs": n,
        **counts,
        **{f"{o}_rate": round(counts[o] / n, 3) for o in OUTCOMES},
        "tool_errors": sum(r["tool_errors"] for r in records),
        "crashed_runs": sum(r["error"] is not None for r in records),
        "unverified_smiles_warnings": sum(len(r["warnings"]) for r in records),
        "avg_tool_calls": round(mean(len(r["tools"]) for r in records), 2),
        "avg_latency_s": round(mean(r["latency_s"] for r in records), 2),
        "avg_tokens": round(mean(r["input_tokens"] + r["output_tokens"] for r in records)),
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in records),
    }


def format_report(entries: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
    lines = [
        f"{'id':<5}{'category':<14}{'comp/part/inc':<15}{'tool err':<9}{'latency':<9}{'tokens':<8}",
    ]
    for entry in entries:
        subset = [r for r in records if r["id"] == entry["id"]]
        if not subset:
            continue
        s = summarise(subset)
        triple = f"{s['complete']}/{s['partial']}/{s['incorrect']}"
        lines.append(
            f"{entry['id']:<5}{entry['category']:<14}{triple:<15}{s['tool_errors']:<9}"
            f"{s['avg_latency_s']:<9}{s['avg_tokens']:<8}"
        )
    total = summarise(records)
    lines.append("-" * 60)
    lines.append(
        f"OVERALL  complete {total['complete_rate']:.0%}  partial {total['partial_rate']:.0%}  "
        f"incorrect {total['incorrect_rate']:.0%}  ({total['runs']} runs)"
    )
    lines.append(
        f"tool-call errors {total['tool_errors']}, crashed runs {total['crashed_runs']}, "
        f"unverified-SMILES warnings {total['unverified_smiles_warnings']}, "
        f"avg tool calls/run {total['avg_tool_calls']}, avg latency {total['avg_latency_s']}s, "
        f"avg tokens/run {total['avg_tokens']}"
    )
    return "\n".join(lines)


def build_settings(mock_tools: bool) -> Settings:
    base = get_settings()
    update: dict[str, Any] = {"trace_path": PROJECT_ROOT / "logs" / "eval_trace.jsonl"}
    if mock_tools:
        update["pubchem_mode"] = "offline"
    return base.model_copy(update=update)


async def main_async(args: argparse.Namespace) -> None:
    entries = load_golden(args.file)
    if args.ids:
        wanted = set(args.ids.split(","))
        entries = [e for e in entries if e["id"] in wanted]
    if args.category:
        entries = [e for e in entries if e["category"] == args.category]
    if not entries:
        raise SystemExit("No golden entries selected.")

    settings = build_settings(args.mock_tools)
    tag = datetime.now(UTC).strftime("%H%M%S")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    async with open_assistant(settings) as assistant:
        for entry in entries:
            for run in range(1, args.runs + 1):
                record = await run_once(assistant, entry, run, tag)
                records.append(record)
                print(
                    f"{entry['id']} run {run}: {record['outcome']:<9} tools={record['tools']} "
                    f"{record['latency_s']}s"
                    + (f"  ERROR {record['error']}" if record["error"] else ""),
                    flush=True,
                )
                if args.pause:
                    await asyncio.sleep(args.pause)

    print()
    print(format_report(entries, records))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = ("-mock-tools" if args.mock_tools else "") + (
        "-reinvent" if settings.enable_reinvent else ""
    )
    out = RESULTS_DIR / f"{stamp}{suffix}.json"
    out.write_text(
        json.dumps(
            {
                "model": settings.gemini_model,
                "mock_tools": args.mock_tools,
                "enable_reinvent": settings.enable_reinvent,
                "runs_per_question": args.runs,
                "wall_time_s": round(time.perf_counter() - started, 1),
                "summary": summarise(records),
                "by_category": {
                    c: summarise([r for r in records if _category(entries, r["id"]) == c])
                    for c in sorted({e["category"] for e in entries})
                },
                "per_question": {
                    e["id"]: summarise([r for r in records if r["id"] == e["id"]]) for e in entries
                },
                "records": records,
            },
            indent=1,
        )
    )
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


def _category(entries: list[dict[str, Any]], entry_id: str) -> str:
    return next(e["category"] for e in entries if e["id"] == entry_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniChemAgent on the golden suite.")
    parser.add_argument("--runs", type=int, default=3, help="runs per question (default 3)")
    parser.add_argument("--file", type=Path, default=GOLDEN_PATH, help="question file (YAML)")
    parser.add_argument("--ids", help="comma-separated golden ids, e.g. t01,w03")
    parser.add_argument(
        "--category", choices=["tool_specific", "workflow", "boundary", "generated", "complex"]
    )
    parser.add_argument("--mock-tools", action="store_true", help="offline PubChem fixtures")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between runs")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
