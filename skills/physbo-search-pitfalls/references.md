# Benchmark behind the `interval` advice (2026-11-17)

- **Problem:** Branin function on a 30×30 grid (x ∈ [−5, 10], y ∈ [0, 15]), 900 candidates, minimise.
- **Budget:** 40 evaluations (5 uniformly random + 35 Bayesian, score EI, exact GP).
- **Metric:** rank of the best value found among all 900 candidates (0 = global grid optimum).

| Setting | Ranks (seeds 0–5) |
|---|---|
| Single `bayes_search(max_num_probes=35, interval=0)` | 0, 1, 234, 104, 0 (seeds 0–4) |
| Single `bayes_search(max_num_probes=35, interval=1)` | 1, 1, 0, 0, 0, 7 |
| `physbo.api.suggest` loop (re-learns each call), standardised values | 1, 0, 0, 0, 0, 2 |
| `physbo.api.suggest` loop, raw values | 1, 0, 3, 1, 0, 0 |
| Uniform random 40 of 900 | 3, 8, 11, 4, 5, 7 |

PHYSBO 3.1.1 (the first two rows) and 3.2.1 (the `physbo.api` rows), NumPy 2.x, CPU only.
The same advice is recorded in the V-Lab Platform PHYSBO adapter, which uses `interval=1` by default.
