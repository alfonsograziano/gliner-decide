"""Builds on h005, tuned for season 2. It reads the real ball speed cap and speed-up from the game state instead of assuming 820 px/s. It plans each shot against the worse of two guesses about the opponent: it stays where it is, or it goes back to the middle. Tuned on dev seeds 10-29 (not the ladder seeds): a 25 px dead zone won 165-135 against h005, while 32 px lost 143-157, so the dead zone stays at 25."""

import math

NAME = "aim, season 2 tuned"

DEADBAND = 25
MAX_OFF = 20         # px from paddle centre; wider tried at 25 and did not help
DECISION_LAG = 0.15  # s, how late the opponent reacts

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


def reflect(y: float, s: dict) -> float:
    r, span = s["ball_r"], s["court_h"] - 2 * s["ball_r"]
    m = (y - r) % (2 * span)
    return r + (m if m <= span else 2 * span - m)


def landing_y(s: dict) -> float:
    t = (s["my_x"] - s["ball_r"] - s["ball_x"]) / s["ball_vx"]
    return reflect(s["ball_y"] + s["ball_vy"] * t, s)


def aim_point(s: dict) -> float:
    land = landing_y(s)
    speed = min(s["ball_max"], math.hypot(s["ball_vx"], s["ball_vy"]) * s["speedup"])
    half = s["paddle_h"] / 2
    travel = (s["my_x"] - s["ball_r"]) - (s["court_w"] - s["my_x"] + s["ball_r"])
    middle = s["court_h"] / 2
    best_off, best_margin = 0, -1e9
    for off in range(-MAX_OFF, MAX_OFF + 1, 2):
        angle = off / half * 0.9
        vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
        t = travel / vx
        y = reflect(land + vy * t, s)
        reach = max(0.0, t - DECISION_LAG) * s["paddle_speed"] + half + s["ball_r"]
        margin = min(abs(y - s["opp_y"]), abs(y - middle)) - reach - abs(off) * 0.01
        if margin > best_margin:
            best_off, best_margin = off, margin
    return land - best_off


def describe(s: dict) -> str:
    coming = s["ball_vx"] > 0
    target = aim_point(s) if coming else s["court_h"] / 2
    gap = s["my_y"] - target  # positive: target is above the paddle
    if abs(gap) < DEADBAND:
        where = "level with"
    elif abs(gap) < 120:
        where = "slightly above" if gap > 0 else "slightly below"
    else:
        where = "far above" if gap > 0 else "far below"
    if coming:
        return f"The aim point is {where} your paddle."
    return f"The waiting spot is {where} your paddle."
