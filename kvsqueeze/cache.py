"""A minimal KV cache you can actually watch evict tokens.

A real KV cache stores key/value tensors per token per layer. None of that
matters for reasoning about eviction policy: what matters is which token
positions survive and which get dropped. So a "token" here carries just enough
to model that decision: its position, a small integer id (stand-in for content),
and a running record of how much attention it has received.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Token:
    position: int
    token_id: int
    # cumulative attention mass this token has received from later queries;
    # heavy-hitter policies keep the tokens with the highest values here
    attention_mass: float = 0.0
    is_evicted: bool = False


@dataclass
class KVCache:
    """Holds tokens and enforces a hard budget via an eviction policy.

    ``budget`` is the maximum number of live (non-evicted) tokens allowed. When
    an append would exceed it, the policy decides who gets dropped.
    """

    budget: int
    tokens: list[Token] = field(default_factory=list)
    evictions: int = 0

    def live(self) -> list[Token]:
        return [t for t in self.tokens if not t.is_evicted]

    def size(self) -> int:
        return len(self.live())

    def append(self, token: Token, policy) -> None:
        self.tokens.append(token)
        if self.size() > self.budget:
            victim = policy.choose_victim(self.live())
            if victim is not None:
                victim.is_evicted = True
                self.evictions += 1

    def record_attention(self, positions: list[int], weight: float = 1.0) -> None:
        """A query attended to these positions; credit them.

        Heavy-hitter policies use this signal to decide what to keep.
        """
        wanted = set(positions)
        for t in self.tokens:
            if t.position in wanted and not t.is_evicted:
                t.attention_mass += weight

    def has(self, position: int) -> bool:
        for t in self.tokens:
            if t.position == position:
                return not t.is_evicted
        return False
