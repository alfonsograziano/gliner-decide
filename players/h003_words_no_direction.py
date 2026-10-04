"""Same as h002, but the text no longer says whether the ball is coming or going. In the browser test the phrase "moving away" made the model pick move_down even when the ball was above, so this removes the confusing words and keeps only where the ball is."""

NAME = "words, no direction"

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
    if abs(gap) < DEADBAND:
        where = "level with"
    elif abs(gap) < 120:
        where = "slightly above" if gap > 0 else "slightly below"
    else:
        where = "far above" if gap > 0 else "far below"
    return f"The ball is {where} your paddle."
