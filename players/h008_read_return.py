"""Builds on h006 and reads the opponent's return before it happens. A trace of lost points showed that almost every point is lost at the top or bottom wall by less than 3 px: the paddle moves the right way but arrives too late. Just before the opponent hits, its paddle is already in place, and where the ball meets that paddle sets the return angle. So while the ball is going away, this harness works out where the ball will meet the opponent, reads the opponent's paddle position, follows the return shot back to your side, and waits there. When the ball is still far from the opponent (more than ANTICIPATE_S seconds away), the opponent may still be moving, so it waits in the middle as before. Dev result (seeds 10-19, not on the ladder): lost 49-101 to h005, and 56-94 at best across ANTICIPATE_S values. Not sent to the ladder. Lesson: the opponent paddle keeps shifting by up to 30 px as the ball arrives, which moves the return by more than half a radian, so reading it early sends the paddle to the wrong place."""

import math

NAME = "read the opponent's return"

DEADBAND = 32
MAX_OFF = 20         # px from paddle centre; with a 32 px dead zone, more than ~21 risks missing the ball
DECISION_LAG = 0.15  # s, how late the opponent reacts
ANTICIPATE_S = 0.5   # s before the opponent's hit from which we trust its paddle position

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


def waiting_spot(s: dict) -> float:
    """Where the opponent's return will reach your side, read from where its paddle is now."""
    r, middle = s["ball_r"], s["court_h"] / 2
    opp_contact_x = s["court_w"] - s["my_x"] + r
    t_away = (s["ball_x"] - opp_contact_x) / -s["ball_vx"]
    if t_away > ANTICIPATE_S:
        return middle
    y_opp = reflect(s["ball_y"] + s["ball_vy"] * t_away, s)
    half = s["paddle_h"] / 2
    off = y_opp - s["opp_y"]
    if abs(off) > half + r:
        return middle  # the opponent is going to miss
    speed = min(s["ball_max"], math.hypot(s["ball_vx"], s["ball_vy"]) * s["speedup"])
    angle = off / half * 0.9
    t = ((s["my_x"] - r) - opp_contact_x) / (math.cos(angle) * speed)
    return reflect(y_opp + math.sin(angle) * speed * t, s)


def describe(s: dict) -> str:
    coming = s["ball_vx"] > 0
    target = aim_point(s) if coming else waiting_spot(s)
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
