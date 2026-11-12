"""Streamlit puts ui/ first on sys.path, so `app` there is ui/app.py, not the API package."""

import ast
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def test_ui_does_not_import_the_api_package():
    offenders = []
    for path in UI_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            offenders += [f"{path.name}: {n}" for n in names if n == "app" or n.startswith("app.")]
    assert not offenders, offenders
