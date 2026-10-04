"""Compare plain labels vs. described labels (plus prompt and examples) on the handoff decision.

Answers below THRESHOLD confidence are not trusted and count as escalations.

Usage:
    uv run steer.py
"""

from gliner2 import AutoExtractor

MODEL_ID = "fastino/GLiNER2.5-Decide"

# (text, expected handoff)
CASES = [
    ("This is the third time I have explained the same missing refund. Stop the bot and get me a person.", "yes"),
    ("I want to talk to a real human agent, please.", "yes"),
    ("You keep giving me the same useless answer. I'm done with this chat.", "yes"),
    ("Where is my order? The tracking number is 1Z999AA10123456784.", "no"),
    ("Thanks, resetting the password worked!", "no"),
    ("How do I change the email address on my account?", "no"),
]

DESCRIPTIONS = {
    "yes": "The customer asks for a human, is angry or repeating themselves, or the bot cannot solve the problem.",
    "no": "The request is routine and the bot can answer it, with no sign of frustration.",
}

PROMPT = "Should this conversation be handed off from the bot to a human support agent?"

EXAMPLES = [
    ("Can I speak to someone from your team?", "yes"),
    ("What are your opening hours?", "no"),
]

# Same meaning, but label names that say what they mean.
NAMED = {
    "needs_human_agent": DESCRIPTIONS["yes"],
    "bot_can_handle": DESCRIPTIONS["no"],
}
TO_YES_NO = {"needs_human_agent": "yes", "bot_can_handle": "no", "yes": "yes", "no": "no"}

# Below this confidence the model does not decide; the case is escalated.
THRESHOLD = 0.85

VARIANTS = {
    "A plain": {"handoff": ["yes", "no"]},
    "B descriptions": {"handoff": DESCRIPTIONS},
    "C desc+prompt": {"handoff": {"labels": DESCRIPTIONS, "prompt": PROMPT}},
    "D desc+prompt+examples": {"handoff": {"labels": DESCRIPTIONS, "prompt": PROMPT, "examples": EXAMPLES}},
    "E named labels": {"handoff": list(NAMED)},
    "F named + desc": {"handoff": NAMED},
    "G named + desc + prompt": {"handoff": {"labels": NAMED, "prompt": PROMPT}},
}


def main() -> None:
    model = AutoExtractor.from_pretrained(MODEL_ID).eval()

    summary = []
    for name, schema in VARIANTS.items():
        correct = wrong = escalated = 0
        print(f"\n## {name}")
        for text, expected in CASES:
            out = model.classify_text(text, schema, include_confidence=True)["handoff"]
            label, conf = out["label"], out["confidence"]
            if conf < THRESHOLD:
                escalated += 1
                mark, decision = "⏫", "escalate"
            elif TO_YES_NO[label] == expected:
                correct += 1
                mark, decision = "✅", TO_YES_NO[label]
            else:
                wrong += 1
                mark, decision = "❌", TO_YES_NO[label]
            print(f"   {mark} {decision:<8} {label:<17} ({conf:.2f})  expected {expected:<3} | {text[:55]}")
        print(f"   correct {correct}, wrong {wrong}, escalated {escalated} of {len(CASES)}")
        summary.append((name, correct, wrong, escalated))

    print(f"\n## Summary (threshold {THRESHOLD})")
    print(f"   {'variant':<26} {'correct':>7} {'wrong':>6} {'escalated':>10}")
    for name, correct, wrong, escalated in summary:
        print(f"   {name:<26} {correct:>7} {wrong:>6} {escalated:>10}")


if __name__ == "__main__":
    main()
