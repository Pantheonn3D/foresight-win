"""Drives the real UI in Chromium against the demo backend and fails on any page error
or broken flow. Run: uv run --with playwright python scripts/ui_smoke.py
(needs a Chromium: `playwright install chromium`, or set CHROMIUM_PATH)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"main": 47930, "setup": 47931, "rec": 47932}


def get(port: int, path: str):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}"))


def start_servers() -> list[subprocess.Popen]:
    procs = []
    for name, port in PORTS.items():
        flag = {"setup": ["--setup"], "rec": ["--recording"]}.get(name, [])
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / "demo_server.py"), "--port", str(port),
                                       *flag]))
    deadline = time.monotonic() + 30
    for port in PORTS.values():
        while True:
            try:
                get(port, "/api/state")
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"demo server on {port} did not start")
                time.sleep(0.3)
    return procs


async def run() -> list[str]:
    failures: list[str] = []

    def check(cond: bool, what: str):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    async with async_playwright() as p:
        kw = {"executable_path": os.environ["CHROMIUM_PATH"]} if os.environ.get("CHROMIUM_PATH") else {}
        browser = await p.chromium.launch(**kw)
        for width in (1440, 375):
            page = await browser.new_page(viewport={"width": width, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            main, setup, rec = (f"http://127.0.0.1:{PORTS[k]}/" for k in ("main", "setup", "rec"))

            await page.goto(setup)
            await page.wait_for_selector("text=to download once")
            check(await page.is_enabled("[data-act=start-setup]"), f"{width}: setup shows sizes and a download button")

            await page.goto(main + "#/settings")
            await page.wait_for_selector("#f-language")
            lang = "Swedish" if width == 1440 else "Norwegian"
            await page.select_option("#f-language", lang)
            await page.wait_for_timeout(400)
            check(get(PORTS["main"], "/api/state")["settings"]["language"] == lang, f"{width}: settings change is saved")

            await page.goto(main)
            await page.wait_for_selector(".m-item", state="attached")
            before = len(get(PORTS["main"], "/api/meetings"))
            victim = "Interview" if width == 1440 else "Group standup"
            await page.evaluate(f"[...document.querySelectorAll('.m-item')].find(b => b.textContent.includes('{victim}')).click()")
            await page.wait_for_selector("[data-act=delete-meeting]")
            await page.click("[data-act=delete-meeting]")
            await page.click("#dialog [value=cancel]")
            await page.wait_for_timeout(300)
            check(len(get(PORTS["main"], "/api/meetings")) == before, f"{width}: cancel in the delete dialog keeps the meeting")
            await page.click("[data-act=delete-meeting]")
            await page.click("#dialogOk")
            await page.wait_for_timeout(600)
            check(len(get(PORTS["main"], "/api/meetings")) == before - 1, f"{width}: delete dialog deletes the meeting")

            await page.goto(main)
            await page.wait_for_selector(".m-item", state="attached")
            await page.evaluate("[...document.querySelectorAll('.m-item')].find(b => b.textContent.includes('P5')).click()")
            await page.wait_for_selector("#enhanced")
            check("Action items" in await page.text_content("#enhanced"), f"{width}: enhanced notes render")
            await page.fill(".askbar input", "When is the ethics form due?")
            await page.click(".askbar button")
            await page.wait_for_function("document.querySelector('#mAnswer')?.textContent.includes('[1]')", timeout=10000)
            check(True, f"{width}: meeting question streams an answer")

            await page.goto(rec)
            await page.wait_for_selector(".m-item", state="attached")
            await page.evaluate("document.querySelectorAll('.m-item')[0].click()")
            await page.wait_for_selector("#timer")
            t1 = await page.text_content("#timer")
            await page.wait_for_timeout(2200)
            t2 = await page.text_content("#timer")
            check(t1 != t2, f"{width}: live events reach the page over the WebSocket ({t1} -> {t2})")
            await page.fill("#notes", "- scope: human-AI handoff\n- deadline?\n- typed during the test")
            await page.wait_for_timeout(800)
            mid = get(PORTS["rec"], "/api/state")["recording"]["meeting_id"]
            check("typed during the test" in get(PORTS["rec"], f"/api/meetings/{mid}")["notes"],
                  f"{width}: notes autosave")

            check(not errors, f"{width}: no page or console errors {errors}")
            await page.close()
        await browser.close()
    return failures


def main() -> None:
    procs = start_servers()
    try:
        try:
            failures = asyncio.run(run())
        except Exception as e:
            failures = [f"crashed: {e}"]
            print(f"FAIL crashed: {e}")
    finally:
        for p in procs:
            p.terminate()
    print(f"\n{len(failures)} failure(s)" if failures else "\nUI smoke test passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
