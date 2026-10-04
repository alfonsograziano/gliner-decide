"""Words, but about where the ball will land instead of where it is now. When the ball is coming, the harness follows its path, bounces included, to the point where it reaches your side, and says where that point is compared to your paddle. When the ball is going away, it uses the middle of the court, so the paddle waits in the best spot for the return. That case says "the waiting spot", because the probe showed "the middle of the court is slightly above" was read as stay."""

NAME = "predict landing spot"

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


def landing_y(s: dict) -> float:
    """Height where the ball reaches your paddle, following bounces off the top and bottom walls."""
    r, h = s["ball_r"], s["court_h"]
    t = (s["my_x"] - r - s["ball_x"]) / s["ball_vx"]
    y = s["ball_y"] + s["ball_vy"] * t
    span = h - 2 * r
    m = (y - r) % (2 * span)
    return r + (m if m <= span else 2 * span - m)


def describe(s: dict) -> str:
    coming = s["ball_vx"] > 0
    target = landing_y(s) if coming else s["court_h"] / 2
    gap = s["my_y"] - target  # positive: target is above the paddle
    if abs(gap) < DEADBAND:
        where = "level with"
    elif abs(gap) < 120:
        where = "slightly above" if gap > 0 else "slightly below"
    else:
        where = "far above" if gap > 0 else "far below"
    if coming:
        return f"The ball will arrive {where} your paddle."
    return f"The waiting spot is {where} your paddle."
