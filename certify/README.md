# `certify` — exact certification of minimal recurrent behavioral memory

Given an environment whose hidden variable can be enumerated, this tool drives the environment once per hidden
value with a scripted expert, builds the induced finite symbolic model, checks assumptions (A2) and (A4) and the
transitivity of the compatibility relation at every step, and returns the per-step requirements

* `H(G_{E,t} | O_t)` — the instantaneous requirement (behavioral quotient),
* `H(Gamma_t | O_t)` — the minimal recurrent behavioral memory (exact when transitivity holds at every step),
* `H(Gamma^s_t | O_t)` — the strong-congruence upper bound,
* `H(H_t | O_t)` — the full-history information.

It is the pipeline used for the community-benchmark certifications in the paper (bsuite MemoryChain and the
Passive T-maze, environment code unmodified from MIKASA-Base). The exact solver itself is `toy/gamma_solver.py`.

## One command reproduces the paper's certifications

```bash
python -m pip install -r certify/requirements.txt          # numpy, gymnasium, dm-env, matplotlib
python -m certify --paper                                  # 13 certificates, about one second on a laptop
```

`--paper` runs MemoryChain with 1, 2, 3 context bits at memory lengths 10, 30, 100 and the Passive T-maze at
corridor lengths 10, 20, 50, 100, prints one line per benchmark and checks each profile against the hand
derivation (MemoryChain: 0, 0, b, ..., b, 1 bits; T-maze: 0 then 1 bit through the decision). It then solves two
finite instances from the paper that the benchmarks do not exercise: the non-transitive three-history instance of
Appendix A.2, Remark (i) (Gamma is not constructed at t = 2; bracket [0, log2 3]) and the re-reveal gap toy (Gamma
1 bit where the strong congruence charges 2). It exits with status 0 only if every certificate is exact, matches,
and both finite checks pass. Add `--out FILE.jsonl` to keep the benchmark certificates as JSON lines.

### The benchmark environments (not bundled)

The two benchmarks are imported from an **unmodified** MIKASA-Base checkout (MIT licence), which is not included
in this archive. Obtain it and pin the commit used in the paper:

```bash
git clone https://github.com/CognitiveAISystems/MIKASA-Base.git external/MIKASA-Base
git -C external/MIKASA-Base checkout ac81b6f57459150d6d0594b88ed0dd9676acec9e     # release 0.1.0b1, 2025-02-24
```

The default location is `external/MIKASA-Base` next to this package; otherwise pass `--mikasa PATH` or set the
environment variable `MIKASA_BASE`. Only five files of that checkout are imported, and nothing in them is changed
(`sha256sum`, first 16 hex digits):

| file | sha256 |
|---|---|
| `mikasa_base/Passive_T_Maze/env/env_passive_t_maze.py` | `4cac8f32c5bb2b63` |
| `mikasa_base/Bsuite/env/base.py` | `4963b269f4fea2e7` |
| `mikasa_base/Bsuite/env/discounting_chain.py` | `4da92808ae0b82aa` |
| `mikasa_base/Bsuite/env/memory_chain.py` | `750cf1e2b49a54bf` |
| `mikasa_base/Bsuite/env/bsuite_env.py` | `3ae0a0a4f2070d30` |

`gymnasium`, `dm-env` and `matplotlib` in `requirements.txt` are imported by those environment files, not by the
solver; the solver itself needs only `numpy`.

Single benchmarks:

```bash
python -m certify --task memchain --L 30 --bits 2
python -m certify --task tmaze --L 50
python -m certify --task example --L 5        # a dependency-free toy, see adapters/example.py
```

The per-step table looks like this (MemoryChain, L = 10, 2 bits):

```
  t  |hist|  A2  A4 trans   H(G|O)  H(Gam|O)  H(GamS|O)   H(H|O)
  1       4  ok  ok   yes    0.000     0.000      0.000    0.000
  2       4  ok  ok   yes    0.000     0.000      0.000    0.000
  3       4  ok  ok   yes    0.000     2.000      2.000    2.000
 ...
 10       4  ok  ok   yes    0.000     2.000      2.000    2.000
 11       8  ok  ok   yes    1.000     1.000      1.000    2.000

memchain-L10-b2: T=11, hidden values=8
exact: the compatibility relation is transitive at every step; H(Gamma|O) is the minimal recurrent behavioral memory
```

## Certifying your own environment

Write an `Adapter` (see `adapters/example.py` for a complete one) with

| field | meaning |
|---|---|
| `reset(seed) -> obs` | starts an episode and fixes its hidden value |
| `step(action) -> (obs, reward, done)` | one environment step |
| `latent() -> hashable` | the hidden value of the current episode (read from the environment after `reset`) |
| `oracle(t, latent) -> action` | the scripted expert, deterministic given the step and the hidden value |
| `n_latent` | number of distinct hidden values (or `None`: enumeration stops after `patience` episodes without a new value) |
| `symbol(obs) -> hashable` | optional observation-to-symbol map; default: the raw observation rounded to 6 decimals |
| `success(reward) -> bool` | optional; default `reward > 0` |

then

```python
from certify import certify
cert = certify(adapter)
print(cert.table())          # per-step A2 / A4 / transitivity / rates
cert.H_Gamma                 # list of floats, one per step (nan where transitivity fails)
cert.to_dict()               # JSON-serialisable
```

## What the certificate means, and what it does not

* The rates are those of the **induced finite symbolic model** under the recorded expert occupancy, with the
  raw observation (or your `symbol` map) as the observation convention. They are not statements about a
  continuous controller, and they change if the observation convention changes.
* Certification requires (A2), history-determined expert behavior; the public API rejects a model when it fails.
  `H(Gamma_t | O_t)` is the exact minimum only when the compatibility relation is **transitive at every step**
  (the table says `exact:` at the bottom). When transitivity fails, the tool reports the sandwich bracket
  `[H(G|O), H(Gamma^s|O)]` at the failing steps and does **not** certify the minimum. On the non-transitive
  signpost corridor of the paper this tool therefore returns only the loose bracket `[0, 2 + log2 W]`; the matching
  lower and upper bounds that pin those instances to 2 and 1 bits (Appendix A.3) come from separate,
  corridor-specific scripts shipped alongside:

  ```bash
  python isaac/analysis/w4_upper_bound.py --W 1 3 5 7        # certified closed compatible assignment (upper bound; writes a JSON)
  python isaac/analysis/w4_lower_bound.py                    # minimum-entropy colouring lower bound; reads that JSON, reports lower == upper
  python isaac/analysis/w4_bounds_extra.py --W 9 11 13       # both bounds on the wider corridors
  ```
* Requirements on the environment: the hidden variable is fixed within an episode and enumerable; the expert is
  deterministic given `(t, latent)`; the observation sequence is a deterministic function of the hidden value and
  the actions (the tool detects a violation if a revisited hidden value produces a different sequence;
  finite recording cannot establish determinism or complete support coverage by itself). If `n_latent` is
  unknown, the patience rule certifies only the recorded induced model, not exhaustive coverage. Episodes have a
  fixed length. Tasks whose hidden state grows with the episode (e.g. repeat-previous or autoencoding tasks) are
  outside this scope because their hidden values cannot be enumerated at useful lengths.
* Cost: one recorded episode per hidden value plus the exact solver on `hidden values x T` histories; the
  benchmarks above take well under a second each.

## Files

```
certify/
  core.py               Adapter, Certificate, record(), certify()
  cli.py, __main__.py   python -m certify
  adapters/mikasa.py    MemoryChain and Passive T-maze from an unmodified MIKASA-Base checkout
  adapters/example.py   dependency-free template adapter
  finite.py             the two finite instances used by --paper / --finite
  requirements.txt
  results/cert_paper.jsonl   the 13 certificates as produced by --paper
```
