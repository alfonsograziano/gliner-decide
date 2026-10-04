"""Builds on h006 with a smarter waiting spot. Older harnesses wait in the middle of the court while the ball is going away. But after your shot, you already know where it will reach the opponent, and from there the opponent's possible returns only cover part of your side. So this harness tries every return angle the opponent can hit, finds the lowest and highest place the ball could come back to, and waits halfway between them. The rest is h006: built for season 2. It reads the real ball speed cap and speed-up from the game state instead of assuming 820 px/s. It plans each shot against the worse of two guesses about the opponent: it stays where it is, or it goes back to the middle. The dead zone is 32 px instead of 25, half of the 63 px the paddle moves between two decisions, so the paddle stops on the target instead of wobbling around it. Dev result (seeds 10-19, not on the ladder): lost 67-83 to h005 with WAIT_OFF=45. Narrower ranges and blends with the middle never beat plain waiting in the middle (best 76-74, the same as h006). Not sent to the ladder. Lesson: the return angle swings too much with tiny changes in where the ball meets the opponent paddle, so the middle of the court is the safest place to wait."""

import math

NAME = "aim + smart waiting spot"

DEADBAND = 32
MAX_OFF = 20         # px from paddle centre; with a 32 px dead zone, more than ~21 risks missing the ball
DECISION_LAG = 0.15  # s, how late the opponent reacts
WAIT_OFF = 45        # px: widest hit offset we expect from the opponent
WAIT_BLEND = 1.0     # 1 = wait at the predicted spot, 0 = wait in the middle of the court

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
    """Halfway between the lowest and highest place the opponent's return can reach your side."""
    r = s["ball_r"]
    opp_contact_x = s["court_w"] - s["my_x"] + r
    t_away = (s["ball_x"] - opp_contact_x) / -s["ball_vx"]
    y_opp = reflect(s["ball_y"] + s["ball_vy"] * t_away, s)
    speed = min(s["ball_max"], math.hypot(s["ball_vx"], s["ball_vy"]) * s["speedup"])
    half = s["paddle_h"] / 2
    travel = (s["my_x"] - r) - opp_contact_x
    ys = []
    for off in range(-WAIT_OFF, WAIT_OFF + 1, 5):
        angle = off / half * 0.9
        t = travel / (math.cos(angle) * speed)
        ys.append(reflect(y_opp + math.sin(angle) * speed * t, s))
    spot = (min(ys) + max(ys)) / 2
    return WAIT_BLEND * spot + (1 - WAIT_BLEND) * s["court_h"] / 2


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
