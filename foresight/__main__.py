"""Entry point: runs the local API server and shows it in a native window."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

PORT = 47821  # fixed so a second launch notices the first one instead of fighting over the database


def setup_logging(verbose: bool) -> None:
    from . import config

    handlers: list[logging.Handler] = [RotatingFileHandler(config.logs_dir() / "foresight.log", maxBytes=5_000_000,
                                                           backupCount=3, encoding="utf-8")]
    if sys.stderr is not None:  # windowed builds have no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def port_taken(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def serve(port: int) -> "uvicorn.Server":  # noqa: F821
    import uvicorn

    from .server import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning",
                                          log_config=None))
    threading.Thread(target=server.run, name="uvicorn", daemon=True).start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server


class JsApi:
    """Native helpers exposed to the page as window.pywebview.api."""

    def __init__(self):
        self.window = None

    def pick_folder(self):
        import webview

        result = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        return result[0] if result else None


def run_app(port: int, browser: bool) -> None:
    if port_taken(port):
        msg = "Foresight is already running."
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, msg, "Foresight", 0x40)
        else:
            print(msg)
        return
    server = serve(port)
    url = f"http://127.0.0.1:{port}/"
    if browser:
        import webbrowser

        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        import webview

        api = JsApi()
        api.window = webview.create_window("Foresight", url, js_api=api, width=1280, height=820,
                                           min_size=(380, 600), background_color="#F4F4F2", text_select=True)
        from . import config

        webview.start(private_mode=False, storage_path=str(config.data_dir() / "webview"))
    server.should_exit = True
    from .llama import runtime

    runtime.stop()


def main() -> None:
    ap = argparse.ArgumentParser(prog="foresight")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("selftest", help="download models and run the whole pipeline headless")
    st.add_argument("--llm", default="gemma-4-e2b")
    st.add_argument("--wav", required=True)
    st.add_argument("--expect", required=True, help="words that must appear in the transcript")
    st.add_argument("--skip-embed", action="store_true")
    st.add_argument("--backend", default="auto", choices=["auto", "cuda", "vulkan", "cpu"])
    args = ap.parse_args()
    setup_logging(args.verbose)
    if args.cmd == "selftest":
        from .selftest import run

        sys.exit(run(args.llm, args.wav, args.expect, args.skip_embed, args.backend))
    run_app(args.port, args.browser)


if __name__ == "__main__":
    main()
