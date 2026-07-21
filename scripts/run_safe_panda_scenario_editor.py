#!/usr/bin/env python3
"""Serve the dependency-free Safe Panda scenario editor on localhost."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EDITOR_PATH = REPOSITORY_ROOT / "tools" / "scenario_editor" / "index.html"
PLAN_PATH = REPOSITORY_ROOT / "configs" / "safe_panda_core_scenarios_150_plan.json"


class ScenarioEditorHandler(SimpleHTTPRequestHandler):
    """Serve repository files locally with the editor as the root page."""

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        if self.path in {"/", "/scenario-editor", "/scenario-editor/"}:
            self.send_response(302)
            self.send_header("Location", "/tools/scenario_editor/")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[scenario-editor] {self.address_string()} {format % args}")


def validate_editor_assets(root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Validate the static UI and its machine-readable plan before serving."""

    required = (
        root / "tools" / "scenario_editor" / "index.html",
        root / "tools" / "scenario_editor" / "styles.css",
        root / "tools" / "scenario_editor" / "app.js",
        root / "configs" / "safe_panda_core_scenarios_150_plan.json",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"scenario editor files are missing: {', '.join(missing)}")

    payload = json.loads(required[-1].read_text(encoding="utf-8"))
    families = payload.get("scenario_families")
    if not isinstance(families, list) or len(families) != 3:
        raise ValueError("scenario editor requires exactly three scenario families")
    if payload.get("sampling", {}).get("episodes_per_scenario") != 50:
        raise ValueError("scenario editor requires 50 episodes per scenario")
    return {
        "scenario_count": len(families),
        "scenario_ids": [str(item["id"]) for item in families],
        "episodes_per_scenario": 50,
        "total_episodes": int(payload["sampling"]["total_episodes_per_method"]),
    }


def build_server(host: str, port: int, root: Path = REPOSITORY_ROOT) -> ThreadingHTTPServer:
    """Create a localhost HTTP server rooted at the repository."""

    handler = partial(ScenarioEditorHandler, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="serve without opening the default browser")
    parser.add_argument("--check", action="store_true", help="validate editor assets and exit")
    args = parser.parse_args()

    summary = validate_editor_assets()
    if args.check:
        print(json.dumps(summary, indent=2))
        return 0

    server = build_server(args.host, args.port)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/tools/scenario_editor/"
    print(f"Safe Panda Scenario Lab: {url}")
    print("Press Ctrl+C to stop the editor.")

    if not args.no_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping scenario editor.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
