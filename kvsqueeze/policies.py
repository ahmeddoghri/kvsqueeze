"""Eviction policies: the actual research being compared.

Each policy answers one question: given the live tokens and a full cache, which
single token do we drop? Everything else (the benchmark, the metric) stays the
same, so any quality difference is attributable to the policy alone.
"""
from __future__ import annotations

from kvsqueeze.cache import Token


class EvictionPolicy:
    name = "base"

    def choose_victim(self, live: list[Token]) -> Token | None:
        raise NotImplementedError


class NoEviction(EvictionPolicy):
    """The control. Never evicts, so it never forgets. Uses unbounded memory,
    which is exactly the problem everything else is trying to solve."""

    name = "no_eviction"

    def choose_victim(self, live: list[Token]) -> Token | None:
        return None


class RecencyWindowPolicy(EvictionPolicy):
    """Keep the most recent tokens, drop the oldest. Simple, and a surprisingly
    strong baseline right up until the answer depends on something you dropped."""

    name = "recency_window"

    def choose_victim(self, live: list[Token]) -> Token | None:
        if not live:
            return None
        return min(live, key=lambda t: t.position)


class HeavyHitterPolicy(EvictionPolicy):
    """Keep the tokens that have been attended to the most, drop the least
    attended. This is the H2O idea (Zhang et al., 2023): a small set of tokens
    carries most of the attention mass, so protect those and evict the rest."""

    name = "heavy_hitter"

    def choose_victim(self, live: list[Token]) -> Token | None:
        if not live:
            return None
        # drop the least-attended token; break ties by evicting the older one
        return min(live, key=lambda t: (t.attention_mass, t.position))


class AttentionSinkPolicy(EvictionPolicy):
    """StreamingLLM (Xiao et al., 2023): always keep the first few tokens (the
    "attention sinks" that models dump spare attention onto) plus a recent
    window. Evict from the middle. Cheap, and it fixes the collapse that a pure
    recency window hits when it drops the opening tokens."""

    name = "attention_sink"

    def __init__(self, sink_size: int = 4) -> None:
        self.sink_size = sink_size

    def choose_victim(self, live: list[Token]) -> Token | None:
        if not live:
            return None
        ordered = sorted(live, key=lambda t: t.position)
        sinks = {t.position for t in ordered[: self.sink_size]}
        evictable = [t for t in ordered if t.position not in sinks]
        if not evictable:
            # everything alive is a sink; fall back to oldest non-sink slot
            return ordered[0]
        # evict the oldest token that is not a protected sink
        return min(evictable, key=lambda t: t.position)
