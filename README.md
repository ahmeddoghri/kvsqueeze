# 🗜️ kvsqueeze

**How much KV cache can you throw away before the model forgets?**

![tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **At 40% of full cache, a heavy-hitter policy holds 57% needle recall while a
> plain recency window is down at 15%.** Same memory, very different memory.
> See the whole curve: `python -m kvsqueeze.eval`.
>
> **Update:** that heavy-hitter win doesn't come from tracking relevance. The
> original policy permanently freezes the moment the cache fills, and never
> accepts another token, so it scores 100% on this benchmark's needle
> **and 0% on recall of anything from the last 15 tokens of a 400-token
> stream, at every budget from 10% to 80%.** The needle-specific attention
> credit the README below used to say it was "built to exploit" turns out to
> change zero of 40 benchmark outcomes if you delete it. Root cause, fix,
> and numbers: `python -m kvsqueeze.eval_v2`.

Long-context inference has exactly one villain and it never RSVPs to the
meeting where you decide it's a problem. The KV cache grows linearly with
sequence length, it never shrinks, and it eats your GPU memory until you
either buy a bigger card or start throwing tokens overboard like it's a
sinking rowboat.

Throwing tokens overboard is the interesting part. But which ones? The dumb
answer, keep the most recent and forget the rest, works great right up until
the question depends on the one thing you already forgot, at which point your
model confidently makes something up instead of admitting it doesn't know.
The research answer is to keep the tokens that actually matter, using signals
like accumulated attention (H2O, Zhang et al. 2023) or protected "attention
sink" tokens (StreamingLLM, Xiao et al. 2023).

kvsqueeze puts those policies side by side on a task where forgetting is
measurable instead of vibes, and shows you exactly how far each one can
compress before recall falls off a cliff. No GPU, no weights, no API keys.
Just the policy logic and a metric that doesn't let anyone lie to you.

---

## The result in one command

```bash
python -m kvsqueeze.eval
```
```
needle-in-a-haystack survival vs cache budget (sequence=400 tokens)

    kept    recency_window  attention_sink    heavy_hitter
     80%              72%             70%            100%
     60%              42%             42%             85%
     40%              15%             12%             57%
     24%               0%              0%             32%
     16%               0%              0%             22%
     10%               0%              0%             12%
```

Read it top to bottom and the story writes itself. Every policy gets the exact
same memory budget at each row, no favoritism. A recency window looks fine at
80% and is flatlining by 24%, because the moment the budget shrinks below the
distance to the needle, the needle ages out and is gone, no goodbye, no error
message, just a model that's forgotten what you told it three paragraphs ago.
Heavy-hitter keeps compressing: it's still recalling at 10% of full cache,
long after the simple policies have given up and gone home.

This is the same shape of result the H2O and StreamingLLM papers report. The
difference is you can reproduce it on a laptop in under a second and read the
policy code in one sitting.

## The benchmark never asks about anything recent, and heavy-hitter exploits that

Every needle in the sweep above lives somewhere in the first 70% of the
stream and gets queried only after the whole 400-token stream finishes. So
the benchmark only ever measures "can you remember something from a while
back," never "can you remember what was just said." A policy that stopped
accepting new tokens the moment its budget filled would still ace it,
because nothing in the benchmark's design would notice.

`HeavyHitterPolicy` does exactly that. It evicts the live token with the
lowest cumulative attention, tie-broken toward the older one. Every token in
this simulation earns its only attention credit from a shared local-recency
mechanism the instant it's near the write head; once that mechanism moves on,
the credit plateaus and never changes again. A token that was *just*
appended hasn't received any of it yet, so it enters at `attention_mass=0.0`,
strictly below every token already through the window. Since eviction runs
immediately after append, the newest arrival is always the unique minimum
once the cache is full, and gets evicted on the spot. Every single time.
Forever. The cache freezes solid after its initial fill.

```bash
python -m kvsqueeze.eval_v2
```
```
recent-needle recall (recall of something from the last 15 tokens of the stream):
  budget    recency_window    attention_sink   heavy_hitter_v1   heavy_hitter_v2
     320               100%              100%                0%              100%
     240               100%              100%                0%              100%
     160               100%              100%                0%              100%
      96               100%              100%                0%              100%
      64               100%              100%                0%              100%
      40               100%              100%                0%              100%
```

0% at every budget from 10% to 80%. Not degraded, not brittle under
pressure, just structurally incapable of ever learning anything new once
full. I went looking for the mechanism the README used to credit for
heavy-hitter's win, the needle-directed attention boost ("later queries
re-attend to it, giving heavy-hitter the signal it's built to exploit"), and
ran the original needle benchmark with that credit deleted entirely.
**Zero of 40 trial outcomes changed, at every budget tested.** The signal
never did any work; the win came entirely from the freeze behavior above,
which happens to look like perfect recall on a benchmark that never asks
about anything after the initial fill.

The real H2O paper doesn't have this failure mode, because it never relies
on this: it splits the cache budget between a heavy-hitter pool and an
unconditionally protected *recent window*, specifically so a new token gets
a grace period to prove whether it matters before eviction pressure applies.
`kvsqueeze/policies_v2.py` adds `HeavyHitterPolicyV2(recent_window=16)`,
restoring that split. It fixes recent recall completely (0% to 100% at
every budget) and, as a bonus, **strictly dominates the original on the
original old-needle axis too** (100% recall at every budget from 10% to
80%, versus the original's 12% to 100%), because a token no longer has to
survive the freeze lottery to become part of the "established" pool.
Confirmed on a held-out set of positions and seeds evaluated exactly once.
`cache.py`/`policies.py`/`eval.py` are untouched, so the published
old-needle curve above still reproduces exactly.

## Install

```bash
git clone https://github.com/ahmeddoghri/kvsqueeze
cd kvsqueeze && pip install -e .
python examples/quickstart.py
```

## Use it

```python
from kvsqueeze.cache import KVCache, Token
from kvsqueeze.policies import HeavyHitterPolicy

cache = KVCache(budget=64)          # keep at most 64 tokens live
policy = HeavyHitterPolicy()        # protect the most-attended tokens

for pos in range(400):
    cache.append(Token(position=pos, token_id=pos), policy)
    cache.record_attention([pos - 1, pos], weight=1.0)   # who got looked at

print(cache.size())        # 64, the budget is never exceeded
print(cache.evictions)     # how many tokens were dropped along the way
```

## The policies

| Policy | What it keeps | The idea |
|---|---|---|
| `NoEviction` | everything | the control, unbounded memory (the problem) |
| `RecencyWindowPolicy` | the newest tokens | simple, strong, until the answer is old |
| `HeavyHitterPolicy` | the most-attended tokens | H2O: a few tokens carry most of the attention |
| `AttentionSinkPolicy` | first few tokens + recent window | StreamingLLM: protect the sinks, stream the rest |

Every policy exposes one method, `choose_victim(live_tokens)`. Write your own in
a dozen lines, drop it into the benchmark, and its survival curve shows up next
to the rest.

## Tests

```bash
pip install pytest && pytest -q      # 19 passing
```

## License

MIT © Ahmed Doghri
