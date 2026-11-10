"""CPU tests for the capstone: the Countdown checker and reward, the problem generator, pass@k, and the
lltRL functions loaded through rl_core. Run:  python test_capstone.py   (or pytest test_capstone.py)."""
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import countdown as cd   # noqa: E402
import passk             # noqa: E402

G = lambda nums, t: json.dumps(dict(nums=nums, target=t))   # noqa: E731


def test_checker_accepts_correct_answers():
    gold = G([3, 7, 25], 46)
    for reply in ["...</think> <answer> 25 + 3 * 7 </answer>",
                  "<answer>(25 + 7*3)</answer>",
                  "<answer> 25 + 3 * 7 = 46 </answer>",                      # "= target" is accepted
                  "<answer> 1 + 1 </answer> no, wait <answer> 7 * 3 + 25 </answer>"]:   # the LAST answer counts
        assert cd.correct(reply, gold) == 1.0, reply


def test_checker_rejects_wrong_or_unsafe_answers():
    gold = G([3, 7, 25], 46)
    cases = {
        "<think>25 + 21 = 46</think>": "no answer tag",
        "<answer> 25 + 21 </answer>": "numbers not used exactly once each",       # 21 is not a given number
        "<answer> 25 + 7 + 3 </answer>": "wrong value",
        "<answer> 25 + 3 * 7 + 3 - 3 </answer>": "numbers not used exactly once each",
        "<answer> 25 + 7 </answer>": "numbers not used exactly once each",         # 3 unused
        "<answer> 25 + 3 * 7 = 47 </answer>": "equals sign with a different right-hand side",
        "<answer> 25 + 3 ** 7 </answer>": "not a valid expression",                # power is not allowed
        "<answer> __import__('os') </answer>": "characters outside digits, + - * / ( )",
        "<answer> 25 + 3 x 7 </answer>": "characters outside digits, + - * / ( )",
        "<answer> -3 + 25 + 7 </answer>": "not a valid expression",                # unary minus is not allowed
        "<answer> 25 / (7 - 7) + 3 </answer>": "division by zero",
        "<answer> 2.5 + 3 </answer>": "characters outside digits, + - * / ( )",
        "<answer> 25 + 3 * 7": "no answer tag",                                    # truncated before the closing tag
    }
    for reply, why in cases.items():
        assert cd.correct(reply, gold) == 0.0, reply
        assert cd.diagnose(reply, gold) == why, (reply, cd.diagnose(reply, gold))
    assert cd.check("6 / (3 - 3)", [6, 3, 3], 2) == (False, "division by zero")


def test_exact_fractions():
    # 4 / 3 * 6 is exactly 8; float arithmetic would still be fine here, but 1/3 chains are not
    assert cd.check("4 / 3 * 6", [4, 3, 6], 8)[0]
    assert cd.check("(1 / 3 + 1 / 3 + 1 / 3) * 9", [1, 3, 1, 3, 1, 3, 9], 9)[0]


def test_format_bonus_alternative():
    gold = G([3, 7, 25], 46)
    assert cd.reward_with_format("<answer> 25 + 7 + 3 </answer>", gold) == 0.1
    assert cd.reward_with_format("no tags at all", gold) == 0.0
    assert cd.reward_with_format("<answer> 25 + 3 * 7 </answer>", gold) == 1.0


def brute_force_values(nums):
    """Every value reachable using all numbers once, by trying every permutation, operator and bracketing."""
    vals = set()
    def build(items):
        if len(items) == 1:
            yield items[0]
            return
        for i in range(1, len(items)):
            for a in build(items[:i]):
                for b in build(items[i:]):
                    for op in "+-*/":
                        if op == "/" and b == 0:
                            continue
                        yield {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
    for perm in itertools.permutations([Fraction(x) for x in nums]):
        vals.update(v for v in build(list(perm)) if v is not None)
    return vals


def test_reachable_matches_brute_force():
    rng = random.Random(1)
    for _ in range(25):
        nums = [rng.randint(1, 12) for _ in range(rng.choice([2, 3, 4]))]
        fast = cd.reachable(tuple(sorted(nums)))
        assert set(fast) == brute_force_values(nums), nums
        for v, expr in list(fast.items())[:20]:                  # every stored expression really reaches its value
            assert cd.check(expr, nums, v)[0] or v.denominator != 1, (nums, v, expr)


def test_generated_problems_are_solvable_and_split_is_disjoint():
    train, test = cd.split_problems(seed=0, n_train=500, n_test=40)
    assert len(train) == 500 and len(test) == 40
    assert sum(len(p["nums"]) == 3 for p in test) == 20
    keys = lambda ps: {(tuple(sorted(p["nums"])), p["target"]) for p in ps}   # noqa: E731
    assert not keys(train) & keys(test)
    for p in train[:200] + test:
        assert cd.check(p["solution"], p["nums"], p["target"])[0], p
        assert cd.correct("<answer> " + p["solution"] + " </answer>", cd.problem_text(p)) == 1.0
    assert cd.split_problems(seed=0, n_train=500, n_test=40) == (train, test)   # same seed, same problems


def test_prompt_round_trip():
    p = dict(nums=[3, 7, 25], target=46, solution="25 + 3 * 7")
    text = cd.prompt(cd.problem_text(p))
    assert "[3, 7, 25]" in text and "equals 46" in text and text.endswith("<think>")


def test_pass_at_k_estimator():
    # 1. k samples drawn without replacement from n: the formula equals brute-force enumeration
    for n, c, k in [(5, 2, 2), (8, 1, 3), (6, 0, 2), (6, 6, 1), (10, 3, 10), (7, 4, 5)]:
        draws = list(itertools.combinations([1] * c + [0] * (n - c), k))
        exact = sum(any(d) for d in draws) / len(draws)
        assert abs(passk.pass_at_k(n, c, k) - exact) < 1e-12, (n, c, k)
    # 2. unbiased: averaged over c ~ Binomial(n, p), the estimate equals 1 - (1 - p)^k exactly
    for n, k, p in [(16, 1, 0.3), (16, 4, 0.05), (32, 8, 0.1), (32, 32, 0.02)]:
        e = sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * passk.pass_at_k(n, c, k) for c in range(n + 1))
        assert abs(e - (1 - (1 - p) ** k)) < 1e-9, (n, k, p, e)
    # 3. the naive estimate (any of the first k correct) is unbiased too but noisier; the formula uses all n
    # 4. pass@1 = c / n
    assert abs(passk.pass_at_k(32, 5, 1) - 5 / 32) < 1e-12


def test_curve_and_ci():
    counts = [0] * 50 + [1] * 30 + [16] * 20
    cur = passk.curve(counts, 16, [1, 2, 4, 8, 16, 32])
    assert 32 not in cur and abs(cur[16] - 0.5) < 1e-12
    assert all(cur[a] <= cur[b] for a, b in [(1, 2), (2, 4), (4, 8), (8, 16)])     # pass@k never falls with k
    ci = passk.curve_with_ci(counts, 16, [1, 16])
    assert ci[16]["lo"] < 0.5 < ci[16]["hi"]
    m, lo, hi = passk.paired_diff_ci([1.0] * 60 + [0.0] * 40, [1.0] * 50 + [0.0] * 50)
    assert abs(m - 0.1) < 1e-12 and lo > 0                                             # 10 points, paired


def test_lltrl_functions_load_and_agree():
    """The lltRL functions we reuse: they load by name, pass_at_k agrees with ours, and the GRPO
    advantages behave as the lab says (zero for a group with equal rewards)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip: torch not installed")
        return
    import rl_core
    src = rl_core.sources()
    assert set(src) == {"kl_k3", "autocast", "group_advantages", "rloo_advantages", "grpo_loss", "completion_logps",
                        "load_policy", "sample", "pass_at_k"}
    ns = rl_core.load("unused", 16, cd.STOPS, tokenizer=None)
    for n, c, k in [(32, 0, 1), (32, 3, 8), (16, 16, 4), (8, 2, 8)]:
        assert abs(ns["pass_at_k"](n, c, k) - passk.pass_at_k(n, c, k)) < 1e-12
    torch = ns["torch"]
    r = torch.tensor([1., 0., 0., 0., 1., 1., 1., 1.])
    adv = ns["group_advantages"](r, 4).view(2, 4)
    assert torch.all(adv[1] == 0) and adv[0, 0] > 0 > adv[0, 1]
    adv_ns = ns["group_advantages"](r, 4, scale=False).view(2, 4)
    assert abs(adv_ns[0, 0].item() - 0.75) < 1e-6
    # grpo_loss with ratio 1: per-sequence mean ("grpo") and constant normalizer ("dr_grpo") differ only in weighting
    lp = torch.zeros(2, 5)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    a = torch.tensor([1.0, 1.0])
    g = ns["grpo_loss"](lp, lp, a, mask, loss_type="grpo").item()             # each reply's mean, then over replies
    d = ns["grpo_loss"](lp, lp, a, mask, loss_type="dr_grpo", max_len=5).item()  # sum over tokens / (2 x 5)
    assert abs(g + 1.0) < 1e-6 and abs(d + 0.6) < 1e-6, (g, d)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")
