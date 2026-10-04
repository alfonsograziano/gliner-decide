"""Builds on h004 and adds offense. Where the ball hits the paddle sets the angle it flies back at. So when the ball is coming, the harness tries hit points from 20 px above the paddle centre to 20 px below, follows each return shot to the other side, and keeps the one that lands furthest out of the opponent's reach. The paddle target becomes the "aim point" that makes that shot, and the text says where the aim point is compared to the paddle. When the ball is going away it waits in the middle, like h004."""

import math

NAME = "aim away from opponent"

DEADBAND = 25
MAX_OFF = 20         # px from paddle centre; bigger hits risk missing the ball
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
    speed = min(820, math.hypot(s["ball_vx"], s["ball_vy"]) * 1.05)
    half = s["paddle_h"] / 2
    travel = (s["my_x"] - s["ball_r"]) - (s["court_w"] - s["my_x"] + s["ball_r"])
    best_off, best_margin = 0, -1e9
    for off in range(-MAX_OFF, MAX_OFF + 1, 5):
        angle = off / half * 0.9
        vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
        t = travel / vx
        y = reflect(land + vy * t, s)
        reach = max(0.0, t - DECISION_LAG) * s["paddle_speed"] + half + s["ball_r"]
        margin = abs(y - s["opp_y"]) - reach - abs(off) * 0.01  # prefer safer hits on ties
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
