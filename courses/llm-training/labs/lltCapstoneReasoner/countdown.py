"""Countdown for RL with verifiable rewards: problem generator, prompt, safe checker and reward.

A problem gives 3 or 4 numbers and a target. A correct answer is an arithmetic expression that uses
every given number exactly once, only + - * / and parentheses, and equals the target exactly.

We generate our own problems because the public Countdown set used by TinyZero
(Jiayi-Pan/Countdown-Tasks-3to4) states no licence (checked 2026-11-07).

Nothing here calls eval() on model output: the checker walks the parsed syntax tree and computes
with exact fractions, so 7 / 2 * 2 is exactly 7 and there is no float rounding.
"""
import ast
import json
import random
import re
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

NL = chr(10)

# ---------------------------------------------------------------- problems

OPS = ("+", "-", "*", "/")


def _apply(a, b, op):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return None if b == 0 else a / b


@lru_cache(maxsize=None)
def reachable(nums):
    """All values reachable by combining EVERY number in the sorted tuple `nums` exactly once.
    Returns a dict value -> one expression string that reaches it."""
    if len(nums) == 1:
        return {Fraction(nums[0]): str(nums[0])}
    out = {}
    idx = range(len(nums))
    seen = set()
    for r in range(1, len(nums)):
        for left in combinations(idx, r):
            lt = tuple(sorted(nums[i] for i in left))
            rt = tuple(sorted(nums[i] for i in idx if i not in left))
            if (lt, rt) in seen:
                continue
            seen.add((lt, rt))
            for a, ea in reachable(lt).items():
                for b, eb in reachable(rt).items():
                    for op in OPS:
                        v = _apply(a, b, op)
                        if v is not None and v not in out:
                            out[v] = "(" + ea + " " + op + " " + eb + ")"
    return out


def make_problems(n, seed, sizes=(3, 4), lo=1, hi=50, tmin=10, tmax=100):
    """n solvable problems. Numbers are drawn uniformly from lo..hi; the target is drawn uniformly
    from the integers in tmin..tmax that the numbers can reach, so every problem has a solution."""
    rng = random.Random(seed)
    out, keys = [], set()
    while len(out) < n:
        k = rng.choice(sizes)
        nums = [rng.randint(lo, hi) for _ in range(k)]
        reach = reachable(tuple(sorted(nums)))
        targets = sorted(int(v) for v in reach if v.denominator == 1 and tmin <= v <= tmax)
        if not targets:
            continue
        t = rng.choice(targets)
        key = (tuple(sorted(nums)), t)
        if key in keys:
            continue
        keys.add(key)
        out.append(dict(nums=nums, target=t, solution=reach[Fraction(t)].strip()[1:-1] if len(nums) > 1 else str(t)))
    return out


def split_problems(seed=0, n_train=20000, n_test=200):
    """Train and test sets with no (numbers, target) pair in both. Test: half 3-number, half 4-number."""
    test = make_problems(n_test // 2, seed + 1000, sizes=(3,)) + make_problems(n_test // 2, seed + 2000, sizes=(4,))
    held = {(tuple(sorted(p["nums"])), p["target"]) for p in test}
    train = [p for p in make_problems(n_train + n_test, seed) if (tuple(sorted(p["nums"])), p["target"]) not in held][:n_train]
    return train, test


# ---------------------------------------------------------------- prompt

TEMPLATE = ("A conversation between User and Assistant. The User gives a puzzle, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process and then gives the answer." + NL +
            "User: Using the numbers {nums}, create an equation that equals {target}. "
            "You can use + - * / and parentheses, and you must use every number exactly once. "
            "Show your reasoning in <think> </think> tags, then give only the equation in <answer> </answer> tags, "
            "for example <answer> (1 + 2) / 3 </answer>." + NL +
            "Assistant: Let me solve this step by step." + NL + "<think>")

# One worked example before the puzzle (a "one-shot" prompt). A small base model rarely writes the answer
# tags from the instruction alone; one example shows the format. The example is not in the train or test sets.
EXAMPLE = ("User: Using the numbers [4, 9, 23], create an equation that equals 59. "
           "You can use + - * / and parentheses, and you must use every number exactly once. "
           "Show your reasoning in <think> </think> tags, then give only the equation in <answer> </answer> tags, "
           "for example <answer> (1 + 2) / 3 </answer>." + NL +
           "Assistant: Let me solve this step by step." + NL +
           "<think> 59 is odd and fairly large. 4 * 9 = 36, and 36 + 23 = 59. That uses 4, 9 and 23 once each. </think>" + NL +
           "<answer> 4 * 9 + 23 </answer>" + NL)
TEMPLATE_ONESHOT = TEMPLATE.replace(NL + "User: ", NL + EXAMPLE + "User: ", 1)
PROMPT_STYLE = "zero"      # "zero" or "oneshot"; train.py and evaluate.py set it from --prompt

# Generation stops at the closing answer tag, or when a base model starts a new turn by itself
STOPS = ["</answer>", NL + "User:"]


def problem_text(p):
    return json.dumps(dict(nums=p["nums"], target=p["target"]))


def prompt(q):
    """q: the problem as a JSON string, so the problem record can travel as one string field."""
    p = json.loads(q)
    t = TEMPLATE_ONESHOT if PROMPT_STYLE == "oneshot" else TEMPLATE
    return t.format(nums=str(p["nums"]), target=p["target"])


# ---------------------------------------------------------------- checker and reward

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
MAX_EXPR_CHARS = 120


def extract_answer(reply):
    """The text inside the LAST complete <answer>...</answer> pair, or None."""
    found = ANSWER.findall(reply)
    return found[-1].strip() if found else None


def _value(node, used):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _value(node.left, used), _value(node.right, used)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise ZeroDivisionError
        return a / b
    raise ValueError("not allowed: " + type(node).__name__)


def check(expr, nums, target):
    """Return (ok, reason). ok is True only if expr uses every number in nums exactly once,
    only + - * / and parentheses, and equals target exactly."""
    if expr is None:
        return False, "no answer tag"
    expr = expr.replace(NL, " ").strip()
    if "=" in expr:                                   # accept "expr = target", nothing else after an equals sign
        left, right = expr.split("=", 1)
        if right.strip() != str(target):
            return False, "equals sign with a different right-hand side"
        expr = left.strip()
    if not expr or len(expr) > MAX_EXPR_CHARS:
        return False, "empty or too long"
    if not re.fullmatch(r"[0-9+\-*/() ]+", expr):
        return False, "characters outside digits, + - * / ( )"
    try:
        tree = ast.parse(expr, mode="eval")
        used = []
        value = _value(tree.body, used)
    except ZeroDivisionError:
        return False, "division by zero"
    except (SyntaxError, ValueError, RecursionError):
        return False, "not a valid expression"
    if sorted(used) != sorted(nums):
        return False, "numbers not used exactly once each"
    if value != target:
        return False, "wrong value"
    return True, "correct"


def correct(reply, gold):
    """Reward used in training: 1.0 for a correct final equation, else 0.0.
    gold: the problem as a JSON string (the same string prompt() receives). Same signature as the
    lltRL lab's GSM8K correct(reply, gold), so the GRPO code there runs unchanged on Countdown."""
    p = json.loads(gold)
    ok, _ = check(extract_answer(reply), p["nums"], p["target"])
    return 1.0 if ok else 0.0


def reward_with_format(reply, gold, bonus=0.1):
    """The TinyZero-style shaped reward: a small bonus for a well-formed but wrong answer. Our reference runs use it (train.py --reward format, the default)."""
    p = json.loads(gold)
    expr = extract_answer(reply)
    ok, why = check(expr, p["nums"], p["target"])
    if ok:
        return 1.0
    return bonus if why in ("wrong value", "numbers not used exactly once each") else 0.0


def diagnose(reply, gold):
    """Why a reply scored what it scored (for failure-case tables)."""
    p = json.loads(gold)
    return check(extract_answer(reply), p["nums"], p["target"])[1]
