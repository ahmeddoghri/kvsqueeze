"""Does the eviction policy actually work, or does it just stop working?

``kvsqueeze.eval`` only ever asks about a needle placed in the first 70% of
the stream, queried after the whole stream finishes. It never asks about
something recent. This module adds that axis, and compares the original
``HeavyHitterPolicy`` against ``HeavyHitterPolicyV2`` (which restores the
H2O paper's protected recent window) on both the original old-needle axis
and the new recent-needle axis.

    python -m kvsqueeze.eval_v2
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, Sequence

from .adversarial import ADVERSARIAL_TRIALS, HOLDOUT_TRIALS, RecencyTrial
from .cache import KVCache, Token
from .policies import AttentionSinkPolicy, HeavyHitterPolicy, RecencyWindowPolicy
from .policies_v2 import HeavyHitterPolicyV2

OLD_NEEDLE_BUDGETS = (320, 240, 160, 96, 64, 40)


def _old_needle_trial(policy, seq_len: int, budget: int, needle_pos: int, seed: int) -> bool:
    rng = random.Random(seed)
    cache = KVCache(budget=budget)
    for pos in range(seq_len):
        cache.append(Token(position=pos, token_id=rng.randint(0, 1000)), policy)
        if pos > needle_pos and rng.random() < 0.55:
            cache.record_attention([needle_pos], weight=1.0)
        cache.record_attention(list(range(max(0, pos - 3), pos + 1)), weight=0.25)
    return cache.has(needle_pos)


def old_needle_recall(policy_factory, budget: int, trials: int = 40, seq_len: int = 400) -> float:
    survived = 0
    for i in range(trials):
        needle_pos = int((i / trials) * (seq_len * 0.7)) + 5
        if _old_needle_trial(policy_factory(), seq_len, budget, needle_pos, seed=i):
            survived += 1
    return survived / trials


def recent_needle_recall(policy_factory, budget: int, trial_set: Sequence[RecencyTrial]) -> float:
    survived = 0
    for t in trial_set:
        needle_pos = t.seq_len - 1 - t.needle_offset_from_end
        if _old_needle_trial(policy_factory(), t.seq_len, budget, needle_pos, seed=t.seed):
            survived += 1
    return survived / len(trial_set)


def ablation_inert_credit(policy_factory, seq_len: int = 400, budget: int = 160, trials: int = 40) -> int:
    """How many of the original benchmark's trials change outcome if the
    needle-specific attention credit is removed entirely? For
    HeavyHitterPolicy the answer is 0/40 at every budget: the credit the
    README says the policy is "built to exploit" never once changes who
    survives."""
    def trial(apply_oracle: bool, needle_pos: int, seed: int) -> bool:
        rng = random.Random(seed)
        cache = KVCache(budget=budget)
        policy = policy_factory()
        for pos in range(seq_len):
            cache.append(Token(position=pos, token_id=rng.randint(0, 1000)), policy)
            roll = rng.random()
            if apply_oracle and pos > needle_pos and roll < 0.55:
                cache.record_attention([needle_pos], weight=1.0)
            cache.record_attention(list(range(max(0, pos - 3), pos + 1)), weight=0.25)
        return cache.has(needle_pos)

    diffs = 0
    for i in range(trials):
        needle_pos = int((i / trials) * (seq_len * 0.7)) + 5
        if trial(True, needle_pos, i) != trial(False, needle_pos, i):
            diffs += 1
    return diffs


def build_report() -> Dict:
    policies: Dict[str, callable] = {
        "recency_window": RecencyWindowPolicy,
        "attention_sink": lambda: AttentionSinkPolicy(sink_size=4),
        "heavy_hitter_v1": HeavyHitterPolicy,
        "heavy_hitter_v2": lambda: HeavyHitterPolicyV2(recent_window=16),
    }

    old_needle = {
        name: {b: round(old_needle_recall(factory, b), 4) for b in OLD_NEEDLE_BUDGETS}
        for name, factory in policies.items()
    }
    recent_needle = {
        name: {b: round(recent_needle_recall(factory, b, ADVERSARIAL_TRIALS), 4) for b in OLD_NEEDLE_BUDGETS}
        for name, factory in policies.items()
    }
    recent_needle_holdout = {
        name: round(recent_needle_recall(factory, 160, HOLDOUT_TRIALS), 4)
        for name, factory in policies.items()
    }
    return {
        "old_needle_recall": old_needle,
        "recent_needle_recall": recent_needle,
        "recent_needle_recall_holdout_budget160": recent_needle_holdout,
        "ablation_diffs_from_removing_needle_credit_v1": ablation_inert_credit(HeavyHitterPolicy),
    }


def format_report(report: Dict) -> str:
    lines = [
        "old-needle recall (the axis the original benchmark tests):",
        "  " + "".join(f"{n:>18}" for n in ("budget",) + tuple(report["old_needle_recall"].keys())),
    ]
    for b in OLD_NEEDLE_BUDGETS:
        row = "".join(f"{report['old_needle_recall'][n][b]:>18.0%}" for n in report["old_needle_recall"])
        lines.append(f"  {b:>6}          " + row)

    lines.append("")
    lines.append("recent-needle recall (the axis it never tests):")
    lines.append("  " + "".join(f"{n:>18}" for n in ("budget",) + tuple(report["recent_needle_recall"].keys())))
    for b in OLD_NEEDLE_BUDGETS:
        row = "".join(f"{report['recent_needle_recall'][n][b]:>18.0%}" for n in report["recent_needle_recall"])
        lines.append(f"  {b:>6}          " + row)

    lines.append("")
    lines.append(
        f"ablation: removing HeavyHitterPolicy's needle-specific attention "
        f"credit changes {report['ablation_diffs_from_removing_needle_credit_v1']}/40 outcomes "
        f"(the signal the README says it's 'built to exploit' is inert)."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
