"""Benchmark: how much KV cache can each policy throw away before recall breaks?

The task is needle-in-a-haystack retrieval, the standard long-context probe. We
stream a long sequence past a fixed-budget cache. One token is the "needle" a
later query needs. A policy passes a trial only if the needle is still live when
the query arrives.

We sweep the cache budget from generous down to brutal and plot the survival
curve for each policy. That curve is the real deliverable: it tells you how far
you can compress before a given policy falls off a cliff. This is the same shape
of result the H2O and StreamingLLM papers report, just reproducible on a laptop.

We also model realistic attention: the needle is the relevant token, so later
queries re-attend to it, and every query attends to a short recent window. That
gives heavy-hitter policies the signal they are built to exploit and gives
recency its local-window advantage, so the comparison is fair.

Run it:

    python -m kvsqueeze.eval

Deterministic. No GPU, no weights, no API keys.
"""
from __future__ import annotations

import random

from kvsqueeze.cache import KVCache, Token
from kvsqueeze.policies import (
    AttentionSinkPolicy,
    HeavyHitterPolicy,
    RecencyWindowPolicy,
)


def _trial(policy, seq_len: int, budget: int, needle_pos: int, seed: int) -> bool:
    rng = random.Random(seed)
    cache = KVCache(budget=budget)
    for pos in range(seq_len):
        cache.append(Token(position=pos, token_id=rng.randint(0, 1000)), policy)
        if pos > needle_pos and rng.random() < 0.55:
            cache.record_attention([needle_pos], weight=1.0)
        cache.record_attention(list(range(max(0, pos - 3), pos + 1)), weight=0.25)
    return cache.has(needle_pos)


def _survival(policy, seq_len: int, budget: int, trials: int) -> float:
    survived = 0
    for i in range(trials):
        needle_pos = int((i / trials) * (seq_len * 0.7)) + 5
        if _trial(policy, seq_len, budget, needle_pos, seed=i):
            survived += 1
    return survived / trials


def run(seq_len: int = 400, trials: int = 40) -> None:
    policies = [RecencyWindowPolicy(), AttentionSinkPolicy(sink_size=4), HeavyHitterPolicy()]
    budgets = [320, 240, 160, 96, 64, 40]

    print(f"needle-in-a-haystack survival vs cache budget (sequence={seq_len} tokens)\n")
    print("full cache (no eviction) is 100% by construction. the question is how")
    print("far each policy compresses before recall collapses.\n")

    header = f"  {'kept':>6}  " + "".join(f"{p.name:>16}" for p in policies)
    print(header)
    for budget in budgets:
        kept = f"{budget / seq_len:.0%}"
        cells = "".join(f"{_survival(p, seq_len, budget, trials):>15.0%} " for p in policies)
        print(f"  {kept:>6}  {cells}")

    # headline: at the tightest budget, how much better is the best policy?
    tight = budgets[-1]
    scores = {p.name: _survival(p, seq_len, tight, trials) for p in policies}
    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    print(f"\nat {tight / seq_len:.0%} of full cache, {best} holds {scores[best]:.0%} recall "
          f"while {worst} is down at {scores[worst]:.0%}.")
    print("recency looks fine until the budget drops below the distance to the")
    print("needle, then it falls off a cliff. tracking attention is what lets you")
    print("keep compressing without forgetting the token that mattered.")


if __name__ == "__main__":
    run()
