"""Baseline from the browser game's Numbers mode. The model gets two raw heights and has to compare them itself. Heights are measured from the bottom, so a bigger number is higher up."""

NAME = "numbers baseline"

SCHEMA = {
    "move": {
        "labels": {
            "move_up": "The ball height is a bigger number than the paddle height.",
            "move_down": "The ball height is a smaller number than the paddle height.",
            "stay": "The ball height and the paddle height are almost the same number.",
        },
        "prompt": "Compare the two heights. Which way should the Pong paddle move to block the ball?",
    }
}

ACTIONS = {"move_up": "up", "move_down": "down", "stay": "stay"}


def describe(s: dict) -> str:
    ball_h = round(s["court_h"] - s["ball_y"])
    paddle_h = round(s["court_h"] - s["my_y"])
    toward = "coming toward your paddle" if s["ball_vx"] > 0 else "moving away from your paddle"
    return f"Ball height: {ball_h}. Your paddle height: {paddle_h}. The ball is {toward}."
