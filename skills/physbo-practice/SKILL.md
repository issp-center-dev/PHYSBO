---
name: physbo-practice
description: Use when setting up or running Bayesian optimisation with PHYSBO — choosing between discrete (physbo.search.discrete.Policy) and continuous (physbo.search.range.Policy) search, building the candidate matrix, writing an ask-tell loop with physbo.api.suggest / predict or the physbo-mcp server, choosing score (EI / PI / TS), interval, num_rand_basis, num_random_initial, or deciding what to record for reproducibility.
---

# PHYSBO in practice

PHYSBO proposes **where to evaluate next**. It does not run your simulation or experiment.
The usual pattern is an ask-tell loop: you ask PHYSBO for a candidate, evaluate it yourself, and tell PHYSBO the result.

## 1. Pick the search type

| Situation | Use |
|---|---|
| Finite set of candidates (materials list, grid, enumerated structures) | `physbo.search.discrete.Policy(test_X=X)` or `physbo.api` |
| Continuous box `min_X <= x <= max_X` | `physbo.search.range.Policy(min_X=..., max_X=...)` |
| Several objectives | `discrete_multi` / `range_multi` (Pareto front) |

`physbo.api` (and the MCP server) cover **single-objective discrete search** only.

## 2. Ask-tell with the JSON API (recommended for tools and agents)

```python
from physbo import api

req = {"space": [{"name": "U", "values": [0, 2, 4, 6, 8]},
                 {"name": "n", "values": [0.6, 0.8, 1.0]}],
       "direction": "minimize", "seed": 0, "observations": []}
while len(req["observations"]) < budget:
    r = api.suggest(req)                      # {"status": "ok", "mode": "random"|"bayes", "suggestions": [...]}
    for s in r["suggestions"]:
        y, ok = my_simulation(**s["params"])  # your code
        if ok:                                # pass only valid results
            req["observations"].append({"index": s["index"], "value": y})
    req["seed"] += 1
```

- The first `num_random_initial` suggestions (default 5) are uniformly random. Bayesian suggestions start after that.
- Already-observed candidates are never suggested again. When all candidates are used, `suggestions` is empty.
- `api.predict(req)` gives the surrogate's mean and std on every candidate, in your units and sign convention.
- Every result has `provenance` (PHYSBO/NumPy/SciPy versions, seed, API schema version). Store it with your data.

The MCP server (`pip install "physbo[mcp]"`, then run `physbo-mcp`) exposes the same functions as the tools `get_schema`, `validate_request`, `suggest_next`, `predict_posterior` and `versions`.

## 3. Using `Policy` directly

```python
policy = physbo.search.discrete.Policy(test_X=X)        # X: (n_candidates, n_features) floats
policy.set_seed(0)                                     # seeds the *global* np.random
policy.random_search(max_num_probes=5, simulator=sim)
policy.bayes_search(max_num_probes=30, simulator=sim, score="EI", interval=1)
```

- `simulator(actions)` receives an array of candidate **indices** and must return an array of values. PHYSBO **maximises** them, so return `-loss` to minimise.
- `bayes_search` defaults to `score="TS"`. `EI` and `PI` are also available for single-objective search.
- `simulator=None` returns the chosen actions without evaluating them. This is "interactive mode" (see `tutorial_interactive_mode`).
- `initial_data=(actions, values)` in the constructor seeds the training data.

## 4. Options that matter

| Option | Meaning | Guidance |
|---|---|---|
| `interval` | Re-learn GP hyperparameters every `interval` steps. `0` = only at the first step, `<0` = never | For short runs (tens of evaluations) use `1`; see `physbo-search-pitfalls` |
| `num_rand_basis` | `0` = exact GP; `>0` = random-feature approximation (Bayesian linear model) | Use a few thousand when there are many observations (exact GP training is cubic in the number of observations) |
| `score` | Acquisition: `EI`, `PI`, `TS` | `EI` is a robust default; `TS` is cheap and stochastic |
| feature scaling | Features enter a stationary kernel | Scale each axis to a comparable range (`physbo.api` uses min-max scaling to [0, 1] by default) |

## 5. Record for reproducibility

PHYSBO version, seed, candidate matrix (or `space`), every observation (index, value), score, interval, num_rand_basis and number of random initial points.
