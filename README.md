# 🗜️ kvsqueeze

**How much KV cache can you throw away before the model forgets?**

![CI](https://github.com/ahmeddoghri/kvsqueeze/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **At 40% of full cache, a heavy-hitter policy holds 57% needle recall while a
> plain recency window is down at 15%.** Same memory, very different memory.
> See the whole curve: `python -m kvsqueeze.eval`.

Long-context inference has one villain, and it is the KV cache. It grows
linearly with sequence length, it never shrinks, and it eats your GPU memory
until you either buy a bigger card or start dropping tokens.

Dropping tokens is the interesting option. But which ones? The dumb answer,
keep the most recent and forget the rest, works great right up until the
question depends on something you forgot. The research answer is to keep the
tokens that actually matter, using signals like accumulated attention (H2O,
Zhang et al. 2023) or protected "attention sink" tokens (StreamingLLM, Xiao et
al. 2023).

kvsqueeze puts those policies side by side on a task where forgetting is
measurable, and shows you exactly how far each one can compress before recall
falls off a cliff. No GPU, no weights, no API keys. Just the policy logic and
an honest metric.

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

Read it top to bottom and the story writes itself. Every policy uses the exact
same memory at each row. A recency window looks fine at 80% and is already dead
by 24%, because the moment the budget shrinks below the distance to the needle,
the needle ages out and is gone. Heavy-hitter keeps compressing: it is still
recalling at 10% of full cache, where the simple policies have flatlined.

This is the same shape of result the H2O and StreamingLLM papers report. The
difference is you can reproduce it on a laptop in under a second and read the
policy code in one sitting.

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
pip install pytest && pytest -q      # 7 passing
```

## License

MIT © Ahmed Doghri
