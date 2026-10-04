"""Pong where each paddle is either you or a GLiNER2.5-Decide harness from players/.

The browser runs the game and posts the raw state to /decide a few times a second, once per AI
paddle, with the id of the harness on that side. The server loads that harness the same way
arena.py does, lets it turn the state into text, asks the model, and returns the decision with its
confidence and latency.

Usage:
    uv run pong_server.py                  # http://127.0.0.1:3070
    uv run pong_server.py --port 3071      # or set PORT
    uv run pong_server.py --host 0.0.0.0   # expose on the network (there is no auth)
    uv run pong_server.py --device cpu     # default is auto: cuda, then Apple mps, then cpu
"""

import argparse
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import arena

HERE = Path(__file__).parent
BROWSER_SEASON = 1  # pong.html uses season 1 physics (ball up to 820 px/s)
MAX_BODY = 64 * 1024


def load_players() -> dict[str, arena.Harness]:
    """Read players/ on every call, so a new harness shows up without a restart."""
    return arena.all_harnesses()


def player_list(players: dict[str, arena.Harness]) -> dict:
    log = arena.read_log()
    champs = {s: arena.champion(log, s) for s in arena.SEASONS}
    played = {hid for m in log for hid in (m["a"], m["b"])}
    out = []
    for hid, h in players.items():
        tags = [f"S{s} champion" for s, c in champs.items() if c == hid]
        if hid not in played:
            tags.append("not played yet")
        out.append({"id": hid, "name": h.name, "idea": h.idea, "tags": tags})
    return {"players": out, "default": champs.get(BROWSER_SEASON) or next(iter(players), None),
            "ballMax": arena.SEASONS[BROWSER_SEASON]["ball_max"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 3070)))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"), help="auto, cpu, mps or cuda")
    args = parser.parse_args()

    brain = arena.Brain(args.device)
    brain.model.classify_text("warm up", {"x": ["a", "b"]})  # so the first move is not slow
    players = load_players()
    lock = threading.Lock()  # one inference at a time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            nonlocal players
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "pong.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(200, {"ok": True, "players": len(players)})
            elif self.path == "/players":
                players = load_players()
                self._json(200, player_list(players))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/decide":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("bad request size")
                req = json.loads(self.rfile.read(length))
                b = req["ball"]
                side = req["side"]  # -1 = left paddle, 1 = right paddle
                if side not in (-1, 1):
                    raise ValueError("side must be -1 or 1")
                state = arena.state_for(
                    side, [float(b["x"]), float(b["y"]), float(b["vx"]), float(b["vy"])],
                    float(req["myY"]), float(req["oppY"]), int(req["scoreMe"]), int(req["scoreOpp"]),
                    float(req["ballMax"]))
                player = req["player"]
            except (KeyError, TypeError, ValueError) as e:
                return self._json(400, {"error": f"bad request: {e}"})

            h = players.get(player)
            if h is None:
                return self._json(404, {"error": f"unknown player {player!r}"})
            try:
                text = h.describe(state)
                h.check_text(text)
                with lock:
                    t0 = time.perf_counter()
                    out = brain.model.classify_text(text, h.schema, include_confidence=True)[h.task]
                    ms = (time.perf_counter() - t0) * 1000
            except ValueError as e:  # the harness leaked a label word
                return self._json(400, {"error": str(e)})
            except Exception:
                traceback.print_exc()
                return self._json(500, {"error": "model failed, see the server log"})
            self._json(200, {"text": text, "label": out["label"], "action": h.actions[out["label"]],
                             "confidence": out["confidence"], "ms": ms})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Pong ready on http://{args.host}:{args.port} with {len(players)} harnesses on {brain.device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
