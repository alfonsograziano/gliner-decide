"""Headless Pong arena: two GLiNER2.5-Decide harnesses play each other.

A harness is one file in players/, named <id>_<slug>.py (for example h002_words.py). It defines:
    NAME     short human name
    SCHEMA   a fixed classify_text schema with exactly one task
    ACTIONS  {label: "up" | "down" | "stay"} for every label in SCHEMA
    describe(state) -> str   turns the game state into the text the model reads

The arena calls the model and moves the paddle. A harness never decides the move itself, and
its text may not contain a label word or up/down/move/stay (see check_text).

Usage:
    uv run arena.py match h001 h002              # play, log, rebuild REPORT.md
    uv run arena.py ladder h003                  # challenge the current champion
    uv run arena.py ladder h002 --vs h001        # first ladder match: name the holder
    uv run arena.py report                       # rebuild REPORT.md from the log
    uv run arena.py list                         # list harnesses
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
PLAYERS = HERE / "players"
LOG = HERE / "results" / "matches.jsonl"
REPORT = HERE / "REPORT.md"
MODEL_ID = "fastino/GLiNER2.5-Decide"

# Physics: same numbers as pong.html, except the ball speed cap, which each season sets.
W, H = 800, 500
PADDLE_W, PADDLE_H, MARGIN = 12, 90, 20
PADDLE_SPEED, BALL_START, BALL_R = 420, 360, 8
SPEEDUP = 1.05               # ball speed grows by 5% on every paddle hit, up to the season's cap

SEASONS = {
    1: {"ball_max": 820, "note": "Same physics as the browser game. Solved by h004: a paddle that predicts the landing spot and waits in the middle never misses, so matches end 0-0."},
    2: {"ball_max": 1600, "note": "The ball can go almost twice as fast, so a well aimed shot can beat a perfect defender."},
}
CURRENT_SEASON = 2
DT = 1 / 120
DECISION_EVERY = 18          # ticks: 0.15 s of game time, like the ~6.5 decisions/s we measured live
POINT_TIMEOUT = 60 / DT      # a rally longer than 60 s is voided
MAX_VOIDS = 20

DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_POINTS = 15          # points per game; odd, so a game cannot tie

BANNED = {"up", "down", "move", "stay"}
ACTION_DIR = {"up": -1, "down": 1, "stay": 0}


# ---------------------------------------------------------------- harness loading

class Harness:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.stem.split("_", 1)[0]
        spec = importlib.util.spec_from_file_location(f"players.{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.name = mod.NAME
        self.idea = (mod.__doc__ or "").strip()
        self.schema = mod.SCHEMA
        self.actions = mod.ACTIONS
        self.describe = mod.describe
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.schema_key = json.dumps(self.schema, sort_keys=True)

        if len(self.schema) != 1:
            raise ValueError(f"{self.id}: SCHEMA must have exactly one task")
        self.task = next(iter(self.schema))
        self.labels = labels_of(self.schema[self.task])
        if set(self.labels) != set(self.actions) or not set(self.actions.values()) <= set(ACTION_DIR):
            raise ValueError(f"{self.id}: ACTIONS must map every label to up/down/stay")
        self.banned = BANNED | {w for l in self.labels for w in re.split(r"[_\W]+", l.lower()) if w}

    def check_text(self, text: str) -> None:
        words = set(re.findall(r"[a-z]+", text.lower()))
        hit = words & self.banned
        if hit:
            raise ValueError(f"{self.id}: describe() leaked answer words {sorted(hit)} in: {text!r}")


def labels_of(task_cfg) -> list[str]:
    if isinstance(task_cfg, dict) and "labels" in task_cfg:
        task_cfg = task_cfg["labels"]
    return list(task_cfg.keys()) if isinstance(task_cfg, dict) else list(task_cfg)


def find_player(ref: str) -> Path:
    p = Path(ref)
    if p.suffix == ".py" and p.exists():
        return p
    matches = sorted(PLAYERS.glob(f"{ref}_*.py"))
    if len(matches) != 1:
        sys.exit(f"Could not find exactly one harness for {ref!r} in {PLAYERS}")
    return matches[0]


# ---------------------------------------------------------------- model

def pick_device(name: str = "auto") -> str:
    """Resolve "auto" to the best device here (cuda, then Apple mps, then cpu); check explicit names."""
    import torch
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if name == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if available[d])
    if not available.get(name):
        sys.exit(f"Device {name!r} is not available here (try --device auto or cpu)")
    return name


class Brain:
    """The model plus a cache. The model is deterministic, so caching by text changes no result.

    The arena stays on the CPU by default so logged results can be reproduced exactly; the
    browser server picks the GPU when there is one.
    """

    def __init__(self, device: str = "cpu"):
        from gliner2 import AutoExtractor
        self.device = pick_device(device)
        print(f"Loading {MODEL_ID} on {self.device}...", flush=True)
        self.model = AutoExtractor.from_pretrained(MODEL_ID).to(self.device).eval()
        self.cache: dict = {}
        self.calls = 0

    def decide(self, h: Harness, text: str) -> str:
        key = (h.schema_key, text)
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.model.classify_text(text, h.schema)[h.task]
        return self.cache[key]


# ---------------------------------------------------------------- game

def state_for(side: int, ball: list, me: float, opp: float, score_me: int, score_opp: int,
              ball_max: float = SEASONS[CURRENT_SEASON]["ball_max"]) -> dict:
    """State from one player's point of view. Every player sees itself on the right side.

    x grows toward your paddle, y grows downward (0 is the top wall).
    """
    x, y, vx, vy = ball
    if side == -1:  # left player: mirror x
        x, vx = W - x, -vx
    return {
        "court_w": W, "court_h": H,
        "ball_x": x, "ball_y": y, "ball_vx": vx, "ball_vy": vy,
        "my_y": me, "opp_y": opp,
        "my_x": W - MARGIN - PADDLE_W, "paddle_h": PADDLE_H, "paddle_speed": PADDLE_SPEED,
        "ball_r": BALL_R, "ball_max": ball_max, "speedup": SPEEDUP,
        "score_me": score_me, "score_opp": score_opp,
    }


def play_game(brain: Brain, left: Harness, right: Harness, seed: int, points: int, ball_max: float,
              trace: list | None = None) -> dict:
    """Play one game. If trace is a list, append one record per point for diagnosis (it does not change play)."""
    rng = random.Random(seed)
    pads = {-1: H / 2, 1: H / 2}
    dirs = {-1: 0, 1: 0}
    score = {-1: 0, 1: 0}
    who = {-1: left, 1: right}
    stats = {s: {"decisions": 0, "labels": Counter(), "texts": set(), "returns": 0} for s in (-1, 1)}
    voids = 0

    def serve(direction: int) -> list:
        a = rng.uniform(-0.4, 0.4)
        return [W / 2, H / 2, math.cos(a) * BALL_START * direction, math.sin(a) * BALL_START]

    ball = serve(rng.choice([-1, 1]))
    tick = point_ticks = 0

    while score[-1] + score[1] < points and voids < MAX_VOIDS:
        if tick % DECISION_EVERY == 0:
            for s in (-1, 1):
                h = who[s]
                text = h.describe(state_for(s, ball, pads[s], pads[-s], score[s], score[-s], ball_max))
                h.check_text(text)
                label = brain.decide(h, text)
                dirs[s] = ACTION_DIR[h.actions[label]]
                st = stats[s]
                st["decisions"] += 1
                st["labels"][label] += 1
                st["texts"].add(text)

        for s in (-1, 1):
            pads[s] = min(H - PADDLE_H / 2, max(PADDLE_H / 2, pads[s] + dirs[s] * PADDLE_SPEED * DT))

        ball[0] += ball[2] * DT
        ball[1] += ball[3] * DT
        if ball[1] < BALL_R:
            ball[1], ball[3] = BALL_R, -ball[3]
        if ball[1] > H - BALL_R:
            ball[1], ball[3] = H - BALL_R, -ball[3]

        for s, face in ((-1, MARGIN + PADDLE_W), (1, W - MARGIN - PADDLE_W)):
            moving_in = ball[2] < 0 if s == -1 else ball[2] > 0
            at_face = (ball[0] - BALL_R <= face and ball[0] > face - PADDLE_W) if s == -1 \
                else (ball[0] + BALL_R >= face and ball[0] < face + PADDLE_W)
            if moving_in and at_face and abs(ball[1] - pads[s]) <= PADDLE_H / 2 + BALL_R:
                off = (ball[1] - pads[s]) / (PADDLE_H / 2)
                speed = min(ball_max, math.hypot(ball[2], ball[3]) * SPEEDUP)
                ball[2] = math.cos(off * 0.9) * speed * -s
                ball[3] = math.sin(off * 0.9) * speed
                ball[0] = face - BALL_R * s
                stats[s]["returns"] += 1

        scored = None
        if ball[0] < -20:
            scored = 1
        elif ball[0] > W + 20:
            scored = -1
        if scored is not None:
            if trace is not None:
                lost = -scored
                trace.append({"loser": who[lost].id, "ball_y_at_miss": round(ball[1]), "paddle_y": round(pads[lost]),
                              "miss_by": round(abs(ball[1] - pads[lost]) - PADDLE_H / 2 - BALL_R),
                              "speed": round(math.hypot(ball[2], ball[3])), "rally_s": round(point_ticks * DT, 1)})
            score[scored] += 1
            ball = serve(-scored)  # serve toward the player who lost the point
            point_ticks = 0
        elif point_ticks > POINT_TIMEOUT:
            voids += 1
            ball = serve(rng.choice([-1, 1]))
            point_ticks = 0

        tick += 1
        point_ticks += 1

    def pack(s):
        st = stats[s]
        return {"points": score[s], "decisions": st["decisions"], "returns": st["returns"],
                "unique_texts": len(st["texts"]), "labels": dict(st["labels"])}

    return {"seed": seed, "left": left.id, "right": right.id, "voids": voids,
            "game_seconds": round(tick * DT, 1), "stats": {left.id: pack(-1), right.id: pack(1)}}


def play_match(brain: Brain, a: Harness, b: Harness, seeds: list[int], points: int, kind: str, season: int) -> dict:
    games = []
    for i, seed in enumerate(seeds):
        left, right = (a, b) if i % 2 == 0 else (b, a)  # swap sides every game
        t0 = time.perf_counter()
        g = play_game(brain, left, right, seed, points, SEASONS[season]["ball_max"])
        g["wall_seconds"] = round(time.perf_counter() - t0, 1)
        games.append(g)
        pa, pb = g["stats"][a.id]["points"], g["stats"][b.id]["points"]
        print(f"  game seed={seed}: {a.id} {pa} - {pb} {b.id}  ({g['wall_seconds']}s, {g['voids']} voids)", flush=True)

    total_a = sum(g["stats"][a.id]["points"] for g in games)
    total_b = sum(g["stats"][b.id]["points"] for g in games)
    winner = a.id if total_a > total_b else b.id if total_b > total_a else None
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "season": season, "kind": kind, "a": a.id, "b": b.id, "points_a": total_a, "points_b": total_b, "winner": winner,
        "seeds": seeds, "points_per_game": points,
        "sha": {a.id: a.sha, b.id: b.sha}, "games": games,
    }


# ---------------------------------------------------------------- log and report

def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    log = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    for m in log:
        m.setdefault("season", 1)  # matches from before seasons existed
    return log


def append_log(entry: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def champion(log: list[dict], season: int):
    """Last ladder winner of the season. A new season starts with the previous season's champion."""
    for s in range(season, 0, -1):
        ladder = [m for m in log if m["season"] == s and m["kind"] == "ladder" and m["winner"]]
        if ladder:
            return ladder[-1]["winner"]
    return None


def all_harnesses() -> dict[str, Harness]:
    out = {}
    for p in sorted(PLAYERS.glob("h*_*.py")):
        h = Harness(p)
        out[h.id] = h
    return out


def season_table(log: list[dict], hs: dict, champ, edited: set) -> list[str]:
    """Leaderboard for one season: Elo over its matches in order, plus record and point share."""
    elo: dict = {}
    rec: dict = {}
    for m in log:
        a, b = m["a"], m["b"]
        for hid in (a, b):
            elo.setdefault(hid, 1000.0)
            rec.setdefault(hid, {"w": 0, "l": 0, "d": 0, "pf": 0, "pa": 0, "m": 0})
        sa = 1.0 if m["winner"] == a else 0.0 if m["winner"] == b else 0.5
        ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
        elo[a] += 32 * (sa - ea)
        elo[b] -= 32 * (sa - ea)
        for hid, pf, pa, res in ((a, m["points_a"], m["points_b"], sa), (b, m["points_b"], m["points_a"], 1 - sa)):
            r = rec[hid]
            r["m"] += 1
            r["pf"] += pf
            r["pa"] += pa
            r["w" if res == 1 else "l" if res == 0 else "d"] += 1

    lines = ["| Rank | Id | Name | Elo | Matches | W-L-D | Points for-against | Point share |",
             "|---:|---|---|---:|---:|---|---|---:|"]
    for i, hid in enumerate(sorted(rec, key=lambda h: -elo[h]), 1):
        r = rec[hid]
        share = r["pf"] / max(1, r["pf"] + r["pa"])
        crown = " 👑" if hid == champ else ""
        flag = " ⚠️ edited" if hid in edited else ""
        name = hs[hid].name if hid in hs else "(file missing)"
        lines.append(f"| {i} | {hid}{crown}{flag} | {name} | {elo[hid]:.0f} | {r['m']} | {r['w']}-{r['l']}-{r['d']} | {r['pf']}-{r['pa']} | {share:.0%} |")
    return lines


def write_report() -> None:
    log = read_log()
    hs = all_harnesses()
    champ = champion(log, CURRENT_SEASON)
    edited = {hid for m in log for hid, sha in m["sha"].items() if hid in hs and hs[hid].sha != sha}

    lines = ["# GLiNER Pong Arena", "",
             "Generated by `arena.py`. Do not edit by hand: run `uv run arena.py report` to rebuild it.", ""]
    if champ:
        lines += [f"**Season {CURRENT_SEASON} champion: {champ} ({hs[champ].name if champ in hs else '?'})**", ""]
    else:
        lines += ["**No champion yet.** Start the ladder with `uv run arena.py ladder <id> --vs <id>`.", ""]

    lines += ["## How it works", "",
              "Two copies of GLiNER2.5-Decide play Pong against each other. Each one has its own harness: the code that turns the game into text, the labels the model picks from, and how each label maps to a paddle move. The model is the same for both, so the only thing that differs is the harness.",
              "",
              f"A match is {len(DEFAULT_SEEDS)} games of {DEFAULT_POINTS} points with fixed seeds, and the players swap sides each game. Whoever scores more points in total wins the match. A rally longer than 60 seconds is voided, and a game stops after {MAX_VOIDS} voids. On the ladder, a challenger plays the current champion and takes the title if it wins. A draw keeps the title where it is.",
              "",
              "Rules that keep the climb honest:",
              "",
              "- The arena calls the model and moves the paddle. A harness only writes text and labels.",
              "- The text may describe the scene, including predictions such as where the ball will arrive. It may not contain a label word or the words up, down, move or stay. The arena rejects the text if it does.",
              "- Both players decide on the same clock, every 0.15 seconds of game time.",
              "- A harness that has played is frozen. A change means a new file with a new id.",
              "",
              "Seasons change the physics. A new season starts with the last champion holding the title.",
              ""]

    for season in sorted(SEASONS, reverse=True):
        slog = [m for m in log if m["season"] == season]
        cfg = SEASONS[season]
        sch = champion(log, season)
        lines += [f"## Season {season}: ball up to {cfg['ball_max']} px/s", "", cfg["note"], ""]
        if not slog:
            lines += [f"No matches yet. Title holder: {sch or 'none'}.", ""]
            continue
        lines += [f"Champion: **{sch}**", ""] + season_table(slog, hs, sch, edited)
        ladder = [m for m in slog if m["kind"] == "ladder"]
        if ladder:
            lines += ["", "**Lineage**", "", "| When | Challenger | Champion | Score | Result |", "|---|---|---|---|---|"]
            for m in ladder:
                res = f"{m['a']} takes the title" if m["winner"] == m["a"] else f"{m['b']} defends" if m["winner"] == m["b"] else "draw, title stays"
                lines.append(f"| {m['time']} | {m['a']} | {m['b']} | {m['points_a']}-{m['points_b']} | {res} |")
        lines.append("")

    lines += ["## All matches", "",
              "| When | Season | Kind | A | B | Score | Winner | Returns A / B | Voids | Unique texts A / B |",
              "|---|---:|---|---|---|---|---|---|---:|---|"]
    for m in reversed(log):
        ra = sum(g["stats"][m["a"]]["returns"] for g in m["games"])
        rb = sum(g["stats"][m["b"]]["returns"] for g in m["games"])
        voids = sum(g["voids"] for g in m["games"])
        ua = max(g["stats"][m["a"]]["unique_texts"] for g in m["games"])
        ub = max(g["stats"][m["b"]]["unique_texts"] for g in m["games"])
        lines.append(f"| {m['time']} | {m['season']} | {m['kind']} | {m['a']} | {m['b']} | {m['points_a']}-{m['points_b']} | {m['winner'] or 'draw'} | {ra} / {rb} | {voids} | {ua} / {ub} |")

    lines += ["", "## Harnesses", ""]
    for hid, h in hs.items():
        lines += [f"### {hid}: {h.name}", "", f"File: [players/{h.path.name}](players/{h.path.name})", "", h.idea or "(no description)", ""]

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}")


def probe(h: Harness, device: str = "cpu") -> None:
    """Sweep ball and paddle positions and both directions; print each distinct text once."""
    brain = Brain(device)
    seen: dict[str, Counter] = {}
    for my_y in range(60, 441, 40):
        for by in range(10, 491, 20):
            for bx in (100, 400, 700):
                for vx, vy in ((400, 150), (400, -150), (-400, 150), (-400, -150)):
                    ball = [bx, by, vx, vy]
                    text = h.describe(state_for(1, ball, my_y, H / 2, 0, 0))
                    h.check_text(text)
                    seen.setdefault(text, Counter())
                    seen[text][brain.decide(h, text)] += 1
    for text, labels in sorted(seen.items(), key=lambda kv: -sum(kv[1].values())):
        label = labels.most_common(1)[0][0]
        print(f"{h.actions[label]:>5}  ({sum(labels.values()):>4}x)  {text}")
    print(f"{len(seen)} distinct texts")


# ---------------------------------------------------------------- CLI

DEVICE_HELP = "cpu (default, reproducible), mps, cuda or auto"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("match", "ladder"):
        p = sub.add_parser(name)
        p.add_argument("a")
        if name == "match":
            p.add_argument("b")
        else:
            p.add_argument("--vs", help="champion to play when the ladder has none yet")
        p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
        p.add_argument("--points", type=int, default=DEFAULT_POINTS)
        p.add_argument("--dry", action="store_true", help="play but do not log")
        p.add_argument("--device", default="cpu", help=DEVICE_HELP)
        p.add_argument("--season", type=int, default=CURRENT_SEASON, choices=sorted(SEASONS))
    sub.add_parser("report")
    sub.add_parser("list")
    p = sub.add_parser("probe", help="show every text a harness writes over a grid of states, and the model's pick")
    p.add_argument("a")
    p.add_argument("--device", default="cpu", help=DEVICE_HELP)
    args = ap.parse_args()

    if args.cmd == "report":
        return write_report()
    if args.cmd == "probe":
        return probe(Harness(find_player(args.a)), args.device)
    if args.cmd == "list":
        for hid, h in all_harnesses().items():
            print(f"{hid}  {h.name}  ({h.path.name})")
        return

    a = Harness(find_player(args.a))
    if args.cmd == "ladder":
        holder = champion(read_log(), args.season) or args.vs
        if not holder:
            sys.exit("No champion yet. Use --vs <id> to name who holds the title.")
        if holder == a.id:
            sys.exit(f"{a.id} is already the champion.")
        b = Harness(find_player(holder))
    else:
        b = Harness(find_player(args.b))

    brain = Brain(args.device)
    print(f"season {args.season} {args.cmd}: {a.id} ({a.name}) vs {b.id} ({b.name})", flush=True)
    m = play_match(brain, a, b, args.seeds, args.points, args.cmd, args.season)
    print(f"Result: {a.id} {m['points_a']} - {m['points_b']} {b.id}  ->  winner: {m['winner'] or 'draw'}  ({brain.calls} model calls)")
    if not args.dry:
        append_log(m)
        write_report()


if __name__ == "__main__":
    main()
