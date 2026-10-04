# GLiNER Pong

Pong in the browser where each paddle is either you or an AI. The AI is [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), a small text classifier, used as a game-playing brain. You can play against it, or pick two AIs and watch them play each other.

The model never sees the game. Each AI has a **harness**: a small Python file that turns the game state into one sentence ("The ball will arrive far above your paddle."), the labels the model chooses from, and how each label maps to a paddle move. The same model sits behind every AI, so the only difference between players is the harness.

<video src="docs/demo.mp4" controls muted width="100%"></video>

*Playing against h004. If the video doesn't load, open [docs/demo.mp4](docs/demo.mp4).*

The repo has two parts:

- **The browser game** (`pong.html` + `pong_server.py`): play it, or watch two harnesses play.
- **The arena** (`arena.py`): a headless tournament that ranks harnesses against each other. Results are in [REPORT.md](REPORT.md).

## Setup

You need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.12 and the dependencies for you
- About 2 GB of free disk space for the model, which downloads from Hugging Face on first run
- A modern browser

```bash
git clone https://github.com/alfonsograziano/gliner-decide.git
cd gliner-decide
uv sync
```

`uv sync` creates `.venv` with Python 3.12 and installs everything pinned in `uv.lock` (PyTorch and `gliner2`). To check that the model loads and answers, run `uv run main.py`. The first run downloads the model, so expect a few minutes.

## Play

```bash
uv run pong_server.py
```

Wait for `Pong ready on http://127.0.0.1:3070`, then open <http://127.0.0.1:3070>. The server loads the model before it starts listening, so on a cold start the page won't open for a few seconds.

On the page:

1. Pick the **left player** and the **right player**. Each one is either **You** or one of the AI harnesses.
   - You vs AI: choose **You** on one side and a harness on the other.
   - AI vs AI: choose a harness on both sides. Pick the same harness twice for a mirror match.
   - Two humans are not supported. If you pick **You** on both sides, the side you didn't just change switches to an AI.
2. Press **Space** (or the **Serve** button) to serve. Space also pauses.
3. Move your paddle with **W**/**S** or **↑**/**↓**. Press **R** (or **Reset**) for a new match.

Changing a player starts a new match. Your choice is remembered in the browser.

Each AI gets a card showing the sentence the model read, the move it chose, its confidence and its latency. The ball in the browser uses season 1 physics (top speed 820 px/s).

### Server options

```bash
uv run pong_server.py --port 3071        # or: PORT=3071 uv run pong_server.py
uv run pong_server.py --host 0.0.0.0     # reachable from other machines
uv run pong_server.py --device cpu       # force the CPU (or: DEVICE=cpu)
```

By default (`--device auto`) the model runs on the GPU when there is one: CUDA, then Apple's MPS, then the CPU. The first line the server prints says which one it picked. On an Apple Silicon Mac, MPS took about 90 ms per decision against about 160 ms on the CPU.

The server has no authentication and runs the model on your machine, so keep the default `127.0.0.1` unless you are on a network you trust.

### Notes on AI vs AI

The model handles one request at a time. With two AIs the requests queue, so each AI decides about half as often as it would alone. Both sides are slowed equally, so the match is fair, but it plays differently from the arena, where every AI decides every 0.15 seconds of game time.

## Add your own AI

Drop a file named `h<number>_<slug>.py` into `players/`. The server picks it up the next time the page loads, with no restart. A harness defines:

| Name | What it is |
|---|---|
| `NAME` | Short name shown in the menu |
| module docstring | Description shown on the AI's card |
| `SCHEMA` | A `classify_text` schema with exactly one task and its labels |
| `ACTIONS` | Maps every label to `"up"`, `"down"` or `"stay"` |
| `describe(state)` | Turns the game state into the text the model reads |

`describe` always sees its own paddle on the right, so one harness plays either side. The text may not contain a label word or the words up, down, move or stay: the server rejects it if it does. See [players/h004_predict_landing.py](players/h004_predict_landing.py) for a small, complete example, and `uv run arena.py probe <id>` to see what a harness writes across many game states.

## The arena

```bash
uv run arena.py list                    # list harnesses
uv run arena.py match h004 h005         # play a match, log it, rebuild REPORT.md
uv run arena.py ladder h006             # challenge the current champion
uv run arena.py match h004 h005 --dry   # play without logging
uv run arena.py report                  # rebuild REPORT.md from the log
```

The arena runs on the CPU unless you pass `--device mps`, `cuda` or `auto`, so logged results stay reproducible. A different device can give tiny numeric differences, which may flip a close call and change a match.

A match is 3 games to 15 points with fixed seeds, and the players swap sides each game. Results are appended to `results/matches.jsonl`, and [REPORT.md](REPORT.md) is generated from that file. Don't edit the report by hand.

## Project layout

| Path | What it is |
|---|---|
| `pong.html` | The browser game |
| `pong_server.py` | Serves the page and answers `/decide` with the model's choice |
| `arena.py` | Harness loading, the headless game, the ladder and the report |
| `players/` | One file per harness |
| `results/matches.jsonl` | Log of every arena match |
| `REPORT.md` | Generated leaderboard and notes on each harness |
| `docs/demo.mp4` | The demo video in this README |
| `main.py`, `steer.py` | Standalone experiments with the model |

## Troubleshooting

- **The page says it can't load the players.** The server isn't running, or you opened `pong.html` directly from disk. Start `uv run pong_server.py` and use the address it prints.
- **`Address already in use`.** Something else has port 3070. Use `--port`.
- **The first start is slow.** The model is downloading (about 2 GB) and then loading. Later starts take a few seconds.
- **A card shows `Model error`.** Check the terminal running the server for the traceback.
