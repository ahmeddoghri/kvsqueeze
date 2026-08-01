"""HeavyHitterPolicy, fixed: give new tokens a chance to be seen before
judging them.

``HeavyHitterPolicy.choose_victim`` evicts the live token with the lowest
``attention_mass``, tie-broken toward the older one. Every token in this
simulation gets its only attention credit from a shared local-recency
mechanism the moment it is near the current write position; once it ages
out of that window its mass plateaus and never changes again unless
something specifically re-attends to it. A token that was *just* appended
has not received any of that credit yet, so it enters the cache at
``attention_mass=0.0``, strictly below every token that already made it
through the window. The eviction check runs immediately after append, so
the newest arrival is the unique minimum and gets evicted on the spot,
every single time, once the cache is full.

The result: the cache freezes solid after its initial fill and never
admits another token again. Measured directly: recall of a needle placed
in the final 15 positions of a 400-token stream is **0% at every tested
budget from 10% to 80% of the sequence**, while a plain recency window and
attention-sink policy both get 100%. This never shows up in the bundled
benchmark because it never asks about anything recent, only about a
needle somewhere in the first 70% of the stream, so a policy that
literally stops working after its first `budget` tokens still scores
perfectly.

The real H2O paper (Zhang et al., 2023) does not have this failure mode
because it never relied on this: it splits the budget between a heavy-hitter
pool and an unconditionally protected *recent window*, precisely so a token
gets a grace period to prove whether it matters before eviction pressure
applies to it. ``HeavyHitterPolicyV2`` restores that split.
"""
from __future__ import annotations

from kvsqueeze.cache import Token
from kvsqueeze.policies import EvictionPolicy


class HeavyHitterPolicyV2(EvictionPolicy):
    name = "heavy_hitter_v2"

    def __init__(self, recent_window: int = 16) -> None:
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        self.recent_window = recent_window

    def choose_victim(self, live: list[Token]) -> Token | None:
        if not live:
            return None
        ordered = sorted(live, key=lambda t: t.position)
        protected = {t.position for t in ordered[-self.recent_window:]}
        evictable = [t for t in ordered if t.position not in protected]
        if not evictable:
            # the whole live set is within the protected recent window;
            # fall back to the oldest slot rather than refuse to evict
            return ordered[0]
        return min(evictable, key=lambda t: (t.attention_mass, t.position))
