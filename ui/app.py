"""Streamlit front end. Talks to the FastAPI backend; run with `streamlit run ui/app.py`."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import requests
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from minichem.config import get_settings
from ui.route_view import show_routes

SETTINGS = get_settings()
REQUEST_TIMEOUT_S = 180
EXAMPLES = [
    "What is the SMILES of ibuprofen?",
    "How many HBD and HBA does CC(=O)Oc1ccccc1C(=O)O have?",
    "Generate analogues of ibuprofen with MW between 180 and 230 and show their SA scores",
]


def call_api(message: str, session_id: str) -> dict:
    """POST /chat. Raises RuntimeError with a readable message on any failure."""
    try:
        response = requests.post(
            f"{SETTINGS.api_url}/chat",
            json={"message": message, "session_id": session_id},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach the API at {SETTINGS.api_url}: {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:300]}")
    return response.json()


def save_feedback(entry: dict, rating: str) -> None:
    """Append one feedback record to the JSONL file."""
    SETTINGS.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "rating": rating,
        "session_id": st.session_state.session_id,
        "question": entry["question"],
        "answer": entry["answer"],
        "tools": [c["tool"] for c in entry["tool_calls"]],
    }
    with SETTINGS.feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def show_molecules(molecules: list[dict]) -> None:
    """Draw each molecule with RDKit, labelled with its handle."""
    parsed = [(m["id"], Chem.MolFromSmiles(m["smiles"])) for m in molecules]
    parsed = [(label, mol) for label, mol in parsed if mol is not None]
    if parsed:
        image = Draw.MolsToGridImage(
            [m for _, m in parsed],
            legends=[i for i, _ in parsed],
            molsPerRow=4,
            subImgSize=(220, 180),
        )
        st.image(image)


def show_steps(tool_calls: list[dict], warnings: list[str]) -> None:
    for number, call in enumerate(tool_calls, 1):
        status = f":red[error: {call['error']}]" if call["error"] else ":green[ok]"
        st.markdown(
            f"**{number}. {call['agent']} / `{call['tool']}`** {status} ({call['duration_ms']} ms)"
        )
        st.json(call["arguments"], expanded=False)
        st.caption(call["result_summary"])
    for warning in warnings:
        st.warning(warning)


def show_turn(index: int, entry: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry.get("routes"):
            show_routes(entry["routes"], entry["molecules"])
        else:
            show_molecules(entry["molecules"])
        with st.expander(f"Agent steps ({len(entry['tool_calls'])} tool calls)"):
            show_steps(entry["tool_calls"], entry["warnings"])
            st.caption(f"{entry['latency_s']}s, {entry['llm_calls']} LLM calls")
        rating = st.feedback("thumbs", key=f"feedback_{index}")
        if rating is not None and index not in st.session_state.rated:
            save_feedback(entry, "up" if rating == 1 else "down")
            st.session_state.rated.add(index)


def main() -> None:
    st.set_page_config(page_title="MiniChemAgent", layout="wide")
    st.title("MiniChemAgent")
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("rated", set())
    with st.sidebar:
        st.caption(f"Session {st.session_state.session_id}")
        st.markdown("**Try:**")
        example = next((e for e in EXAMPLES if st.button(e, key=e)), None)
        if st.button("New conversation"):
            st.session_state.update(session_id=uuid.uuid4().hex[:12], turns=[], rated=set())
            st.rerun()

    for index, entry in enumerate(st.session_state.turns):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        show_turn(index, entry)

    question = st.chat_input("Ask a chemistry question") or example
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Agents working..."):
            try:
                result = call_api(question, st.session_state.session_id)
            except RuntimeError as exc:
                st.error(str(exc))
                return
        st.session_state.turns.append({"question": question, **result})
        st.rerun()


main()
