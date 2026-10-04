"""Baseline from the browser game's Words mode. The harness compares the ball and paddle heights and says it in words (far above, slightly above, level with, slightly below, far below), plus whether the ball is coming or going."""

NAME = "words baseline"

DEADBAND = 25

SCHEMA = {
    "move": {
        "labels": {
            "move_up": "The ball is above the paddle, so the paddle must move up to reach it.",
            "move_down": "The ball is below the paddle, so the paddle must move down to reach it.",
            "stay": "The ball is level with the paddle, so the paddle should stay where it is.",
        },
        "prompt": "Which way should the Pong paddle move to block the ball?",
    }
}

ACTIONS = {"move_up": "up", "move_down": "down", "stay": "stay"}


def describe(s: dict) -> str:
    gap = s["my_y"] - s["ball_y"]  # positive: ball is above the paddle
    toward = "coming toward your paddle" if s["ball_vx"] > 0 else "moving away from your paddle"
    if abs(gap) < DEADBAND:
        where = "level with"
    elif abs(gap) < 120:
        where = "slightly above" if gap > 0 else "slightly below"
    else:
        where = "far above" if gap > 0 else "far below"
    return f"The ball is {where} your paddle and is {toward}."
