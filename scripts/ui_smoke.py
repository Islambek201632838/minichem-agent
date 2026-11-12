"""Browser smoke test of the Streamlit UI (Playwright, Chromium, sync API).

Not part of the project's dependencies and not run by pytest: it needs a running API and UI and
uses the real model. Run it with a Python that has playwright installed (see README, "UI smoke
test"), for example:

    python scripts/ui_smoke.py --ui-url http://127.0.0.1:8601 --retrosynthesis

Checks: the page loads, a real question about imatinib gets an answer with the right molecular
weight (computed independently with RDKit: 493.6), no Streamlit exception or error box is shown,
the agent-steps expander lists the tool calls, thumbs feedback and "New conversation" work. With
--retrosynthesis (the API must run with ENABLE_RETROSYNTHESIS=true) it also asks for a route and
checks that route schemes are drawn. Screenshots go to --out-dir. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PROPERTIES_QUESTION = (
    "What are the molecular weight, logP, HBD and HBA of imatinib? "
    "Look the molecule up by name and use the tools."
)
ROUTE_QUESTION = "Find a retrosynthesis route for ibuprofen and show the steps."
IMATINIB_MW = r"493\.[56]"
ANSWER_TIMEOUT_MS = 300_000
VIEWPORT = {"width": 1400, "height": 2600}
MESSAGE = '[data-testid="stChatMessage"]'
FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    """Print one PASS/FAIL line and remember failures."""
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        FAILURES.append(message)


def page_errors(page: Page) -> list[str]:
    """Text of every Streamlit exception block and error alert currently on the page."""
    found = page.locator('[data-testid="stException"]').all_inner_texts()
    for alert in page.locator('[data-testid="stAlert"]').all():
        if "Cannot reach" in alert.inner_text() or "API error" in alert.inner_text():
            found.append(alert.inner_text())
    return found


def ask(page: Page, question: str) -> str:
    """Send a question, wait until the assistant message with its steps expander is drawn."""
    page.locator('[data-testid="stChatInput"] textarea').fill(question)
    page.keyboard.press("Enter")
    reply = page.locator(MESSAGE).nth(1)
    reply.locator('[data-testid="stExpander"]').first.wait_for(timeout=ANSWER_TIMEOUT_MS)
    return reply.inner_text()


def expand_all(page: Page) -> None:
    """Open every collapsed expander of the last assistant message."""
    for details in page.locator(MESSAGE).last.locator("details").all():
        if details.get_attribute("open") is None:
            details.locator("summary").first.click()
    page.wait_for_timeout(1000)


def test_properties(page: Page, out_dir: Path) -> None:
    text = ask(page, PROPERTIES_QUESTION)
    check(len(text) > 80, f"answer appeared ({len(text)} characters)")
    check(re.search(IMATINIB_MW, text) is not None, "answer contains imatinib MW 493.6")
    check(not page_errors(page), f"no exception or error box on the page: {page_errors(page)}")
    expand_all(page)
    steps = page.locator(MESSAGE).last.inner_text()
    check("calculate_properties" in steps, "agent steps list calculate_properties")
    check("Traceback" not in page.inner_text("body"), "no traceback text anywhere")
    closed = page.locator(MESSAGE).last.locator("details:not([open])").count()
    check(closed == 0, f"agent steps expander is still open at screenshot time (closed: {closed})")
    page.screenshot(path=str(out_dir / "properties.png"), full_page=True)


def test_feedback_and_reset(page: Page) -> None:
    page.locator('[data-testid="stFeedback"] button').first.click()
    page.wait_for_timeout(1500)
    check(not page_errors(page), "thumbs feedback click raised no error")
    page.get_by_role("button", name="New conversation").click()
    page.wait_for_function(f"document.querySelectorAll('{MESSAGE}').length === 0")
    check(True, "New conversation clears the chat")


def test_routes(page: Page, out_dir: Path) -> None:
    text = ask(page, ROUTE_QUESTION)
    check(len(text) > 80, f"route answer appeared ({len(text)} characters)")
    check(not page_errors(page), f"no exception or error box on the page: {page_errors(page)}")
    reply = page.locator(MESSAGE).last
    check("Route 1" in text, "a 'Route 1' heading is drawn")
    expand_all(page)
    check("Forward check" in reply.inner_text(), "a step shows its forward check")
    images = reply.locator("img").count()
    check(images >= 3, f"route molecule and scheme images are present ({images})")
    broken = reply.locator("img").evaluate_all("els => els.filter(e => !e.naturalWidth).length")
    check(broken == 0, f"no broken images ({broken})")
    page.screenshot(path=str(out_dir / "routes.png"), full_page=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    parser.add_argument("--out-dir", type=Path, default=Path("ui_smoke_out"))
    parser.add_argument("--retrosynthesis", action="store_true", help="also test route drawing")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport=VIEWPORT)
        script_errors: list[str] = []
        page.on("pageerror", lambda exc: script_errors.append(str(exc)))
        page.goto(args.ui_url)
        page.locator('[data-testid="stChatInput"]').wait_for(timeout=30_000)
        check(page.title() == "MiniChemAgent", f"page title is {page.title()!r}")
        page.screenshot(path=str(args.out_dir / "start.png"), full_page=True)
        test_properties(page, args.out_dir)
        test_feedback_and_reset(page)
        if args.retrosynthesis:
            test_routes(page, args.out_dir)
        else:
            print("SKIP  route drawing (pass --retrosynthesis with a retrosynthesis-enabled API)")
        check(not script_errors, f"no uncaught JavaScript errors: {script_errors}")
        browser.close()
    print(f"{len(FAILURES)} failed" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
