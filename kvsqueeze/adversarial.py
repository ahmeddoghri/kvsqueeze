"""The needle-recall benchmark never tests recall of anything recent.

``kvsqueeze.eval`` sweeps a needle placed in the first 70% of a 400-token
stream and queries for it only after the stream finishes. That means every
scenario the bundled benchmark ever scores is "can you remember something
from a while ago," never "can you remember what was just said." A policy
that stops accepting new tokens entirely after its first `budget` tokens
would still ace this benchmark, because nothing in it asks about the tail
of the stream.

That is not a hypothetical: ``HeavyHitterPolicy`` does exactly this. See
:mod:`kvsqueeze.policies_v2` for the mechanism. ``ADVERSARIAL_TRIALS`` below
probes the axis the original benchmark never exercises (recall of the most
recent tokens), plus a direct ablation confirming the needle-specific
attention credit the original benchmark's docstring claims is doing work is
actually inert for ``HeavyHitterPolicy``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecencyTrial:
    """Ask for a token from very near the end of the stream, i.e. something
    a real conversation would still expect the cache to remember."""

    seq_len: int
    needle_offset_from_end: int   # needle_pos = seq_len - 1 - offset
    seed: int


ADVERSARIAL_TRIALS: list[RecencyTrial] = [
    RecencyTrial(seq_len=400, needle_offset_from_end=offset, seed=i)
    for i, offset in enumerate(range(0, 15))
]

# A second, disjoint set of seeds and offsets, evaluated exactly once after
# the fix (HeavyHitterPolicyV2's recent_window) was frozen against the
# trials above.
HOLDOUT_TRIALS: list[RecencyTrial] = [
    RecencyTrial(seq_len=400, needle_offset_from_end=offset, seed=1000 + i)
    for i, offset in enumerate(range(0, 15))
]
