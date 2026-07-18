"""Sixty-second tour of kvsqueeze.

    python examples/quickstart.py
"""
from kvsqueeze.cache import KVCache, Token
from kvsqueeze.policies import HeavyHitterPolicy, RecencyWindowPolicy

# Two caches, same tiny budget, different eviction policy.
recency = KVCache(budget=5)
heavy = KVCache(budget=5)
rp, hp = RecencyWindowPolicy(), HeavyHitterPolicy()

# Token 0 is the "needle" we care about. Stream 40 tokens through a 5-slot cache.
for pos in range(40):
    recency.append(Token(position=pos, token_id=pos), rp)
    heavy.append(Token(position=pos, token_id=pos), hp)
    if pos == 0:
        heavy.record_attention([0], weight=50.0)  # the needle gets attended to

print(f"recency still has the needle?      {recency.has(0)}")   # False, aged out
print(f"heavy-hitter still has the needle? {heavy.has(0)}")     # True, protected
print(f"both used the same memory budget:  {recency.budget} tokens")
