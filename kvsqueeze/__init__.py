"""kvsqueeze: how much KV cache can you throw away before the model forgets?

Long-context inference is bottlenecked by the KV cache, which grows linearly
with sequence length and eats your GPU memory alive. The research answer is to
evict tokens you no longer need. The open question is always the same: which
ones, and how many, before the answer degrades?

This package implements the well-known eviction policies (recency window,
attention-score / heavy-hitter, and the StreamingLLM "attention sink + window"
trick) and scores them on a retrieval task where you can actually measure what
was forgotten. No GPU, no weights, just the policy logic and an honest metric.
"""
from kvsqueeze.cache import KVCache, Token
from kvsqueeze.policies import (
    AttentionSinkPolicy,
    EvictionPolicy,
    HeavyHitterPolicy,
    NoEviction,
    RecencyWindowPolicy,
)

__all__ = [
    "KVCache",
    "Token",
    "EvictionPolicy",
    "NoEviction",
    "RecencyWindowPolicy",
    "HeavyHitterPolicy",
    "AttentionSinkPolicy",
]

__version__ = "0.1.0"
