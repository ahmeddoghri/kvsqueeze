"""Tests for the recency-blindness bug in HeavyHitterPolicy and its fix."""

from __future__ import annotations

from kvsqueeze.adversarial import ADVERSARIAL_TRIALS, HOLDOUT_TRIALS
from kvsqueeze.cache import KVCache, Token
from kvsqueeze.eval import run as run_original
from kvsqueeze.eval_v2 import (
    OLD_NEEDLE_BUDGETS,
    ablation_inert_credit,
    old_needle_recall,
    recent_needle_recall,
)
from kvsqueeze.policies import AttentionSinkPolicy, HeavyHitterPolicy, RecencyWindowPolicy
from kvsqueeze.policies_v2 import HeavyHitterPolicyV2

# --- the finding: the needle-specific attention credit is inert ------------

def test_needle_specific_credit_is_inert_for_heavy_hitter():
    """The README claims the needle-directed attention 'gives heavy-hitter
    policies the signal they are built to exploit.' It does not: removing it
    changes zero outcomes."""
    for budget in OLD_NEEDLE_BUDGETS:
        assert ablation_inert_credit(HeavyHitterPolicy, budget=budget) == 0


# --- the finding: HeavyHitterPolicy freezes after its initial fill ---------

def test_original_heavy_hitter_cannot_recall_anything_recent():
    for budget in OLD_NEEDLE_BUDGETS:
        assert recent_needle_recall(HeavyHitterPolicy, budget, ADVERSARIAL_TRIALS) == 0.0


def test_original_heavy_hitter_stops_accepting_new_tokens():
    """Direct mechanism check: once the cache fills, no position past the
    fill point ever survives, at any point later in a long stream."""
    import random

    rng = random.Random(0)
    policy = HeavyHitterPolicy()
    cache = KVCache(budget=160)
    for pos in range(400):
        cache.append(Token(position=pos, token_id=rng.randint(0, 1000)), policy)
        cache.record_attention(list(range(max(0, pos - 3), pos + 1)), weight=0.25)
    live_positions = {t.position for t in cache.live()}
    assert max(live_positions) < 160
    assert not any(p >= 360 for p in live_positions)


def test_recency_and_sink_have_no_such_blind_spot():
    for budget in OLD_NEEDLE_BUDGETS:
        assert recent_needle_recall(RecencyWindowPolicy, budget, ADVERSARIAL_TRIALS) == 1.0
        assert (
            recent_needle_recall(lambda: AttentionSinkPolicy(4), budget, ADVERSARIAL_TRIALS)
            == 1.0
        )


# --- the fix: a protected recent window, matching the real H2O design ------

def test_fixed_heavy_hitter_recalls_recent_content():
    for budget in OLD_NEEDLE_BUDGETS:
        assert (
            recent_needle_recall(lambda: HeavyHitterPolicyV2(16), budget, ADVERSARIAL_TRIALS)
            == 1.0
        )


def test_fixed_heavy_hitter_does_not_regress_old_needle_recall():
    """The fix must not trade away what the original benchmark measured."""
    for budget in OLD_NEEDLE_BUDGETS:
        v1 = old_needle_recall(HeavyHitterPolicy, budget)
        v2 = old_needle_recall(lambda: HeavyHitterPolicyV2(16), budget)
        assert v2 >= v1


def test_fixed_heavy_hitter_protects_a_window_not_the_whole_cache():
    """recent_window=0 must degrade back to the original's tie-break
    behavior rather than silently becoming unbounded."""
    policy = HeavyHitterPolicyV2(recent_window=0)
    assert policy.choose_victim([]) is None


# --- held out, evaluated once ------------------------------------------------

def test_holdout_trials_are_disjoint_from_the_tuning_set():
    adversarial_seeds = {t.seed for t in ADVERSARIAL_TRIALS}
    holdout_seeds = {t.seed for t in HOLDOUT_TRIALS}
    assert not (adversarial_seeds & holdout_seeds)


def test_holdout_confirms_the_fix():
    assert recent_needle_recall(HeavyHitterPolicy, 160, HOLDOUT_TRIALS) == 0.0
    assert recent_needle_recall(lambda: HeavyHitterPolicyV2(16), 160, HOLDOUT_TRIALS) == 1.0


# --- the original benchmark is unaffected -----------------------------------

def test_original_eval_module_untouched():
    import kvsqueeze.policies as policies

    assert not hasattr(policies, "HeavyHitterPolicyV2")


def test_original_benchmark_still_reproduces():
    """python -m kvsqueeze.eval prints the exact published survival curve."""
    assert old_needle_recall(HeavyHitterPolicy, 320) == 1.00
    assert old_needle_recall(HeavyHitterPolicy, 240) == 0.85
    assert old_needle_recall(HeavyHitterPolicy, 160) == 0.575
    assert old_needle_recall(HeavyHitterPolicy, 96) == 0.325
    assert old_needle_recall(HeavyHitterPolicy, 64) == 0.225
    assert old_needle_recall(HeavyHitterPolicy, 40) == 0.125


def test_original_run_executes_without_error():
    run_original(seq_len=400, trials=40)
