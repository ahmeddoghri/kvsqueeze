from kvsqueeze.cache import KVCache, Token
from kvsqueeze.policies import (
    AttentionSinkPolicy,
    HeavyHitterPolicy,
    NoEviction,
    RecencyWindowPolicy,
)


def _fill(policy, n, budget):
    cache = KVCache(budget=budget)
    for pos in range(n):
        cache.append(Token(position=pos, token_id=pos), policy)
    return cache


def test_budget_is_enforced():
    cache = _fill(RecencyWindowPolicy(), 100, budget=10)
    assert cache.size() == 10


def test_no_eviction_keeps_everything():
    cache = _fill(NoEviction(), 50, budget=10)
    assert cache.size() == 50
    assert cache.evictions == 0


def test_recency_keeps_newest():
    cache = _fill(RecencyWindowPolicy(), 20, budget=5)
    live = sorted(t.position for t in cache.live())
    assert live == [15, 16, 17, 18, 19]


def test_attention_sink_protects_first_tokens():
    policy = AttentionSinkPolicy(sink_size=2)
    cache = _fill(policy, 30, budget=6)
    live = {t.position for t in cache.live()}
    # the first two tokens (the sinks) must survive
    assert 0 in live and 1 in live


def test_heavy_hitter_keeps_attended_token():
    policy = HeavyHitterPolicy()
    cache = KVCache(budget=5)
    for pos in range(20):
        cache.append(Token(position=pos, token_id=pos), policy)
        if pos == 0:
            # hammer token 0 with attention right after inserting it
            cache.record_attention([0], weight=100.0)
    assert cache.has(0)  # the heavily-attended token should be protected


def test_recency_drops_old_needle_but_sink_does_not():
    # token 0 is the needle; fill well past the budget
    rec = _fill(RecencyWindowPolicy(), 50, budget=5)
    sink = _fill(AttentionSinkPolicy(sink_size=2), 50, budget=5)
    assert not rec.has(0)   # recency forgot it
    assert sink.has(0)      # sink kept it


def test_eviction_count_matches():
    cache = _fill(RecencyWindowPolicy(), 30, budget=10)
    assert cache.evictions == 20
