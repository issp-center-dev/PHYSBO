---
name: physbo-search-pitfalls
description: Use when a PHYSBO Bayesian optimisation performs no better than random sampling, keeps proposing poor or clustered candidates, or behaves differently between runs — or when you see "WARNING: All actions have already searched.", NaN/inf objective values, huge penalty values for failed evaluations, results that change when another library uses np.random, or very slow bayes_search with many observations.
---

# PHYSBO search pitfalls

Each entry gives the symptom, the cause, and the fix. The measured numbers come from the benchmark in `references.md`.

## Worse than random sampling

**`interval=0` in a long single `bayes_search` call.** `interval=0` learns the GP hyperparameters only at the first Bayesian step (`physbo.search.utility.is_learning`). If that step sees only the few random initial points, the length scales can be badly off for the rest of the run. In the benchmark (Branin, 30×30 grid, 5 random + 35 Bayesian steps, EI), two of five seeds ended at rank 234 and 104 of 900. Uniform random sampling's median is about rank 11.
→ Use `interval=1` (re-learn every step), or call `bayes_search(max_num_probes=1, ...)` in an ask-tell loop. `physbo.api.suggest` learns on every call. With `interval=1` the same benchmark gave ranks 0–7.

**Unscaled features.** One feature in [0, 1000] and another in [0, 1] give a stationary kernel the wrong geometry.
→ Min-max scale each axis (the default in `physbo.api`).

**An objective that is oscillatory or noisy on the candidate grid.** For example, a frequency fit on a 6-point frequency grid: the GP cannot interpolate it, and BO is about as good as random. Nothing is wrong with PHYSBO.
→ Refine the grid, transform the objective (e.g. `log`), or use a local optimiser for fitting.

## Failed evaluations

**NaN or inf values.** These break GP training.
→ Leave failed or non-converged evaluations out. `physbo.api` rejects them with `OBSERVATION_VALUE`.

**Huge penalty values (e.g. 1e12) for failures.** They dominate the GP fit and distort the surrogate everywhere.
→ Leave failures out. If the point must be excluded from future proposals, record it as observed with the worst *valid* value seen so far, and keep it marked as invalid in your own records.

## Reproducibility

**`Policy.set_seed` reseeds the global `np.random`.** Any other code that uses `np.random` changes PHYSBO's random choices, and PHYSBO changes theirs.
→ Save and restore `np.random.get_state()` around PHYSBO calls (`physbo.api` does this), and record the seed.

## Messages

**`WARNING: All actions have already searched.`** Every candidate has been evaluated. The returned history is complete; increase the candidate set if you need more.

## Performance

**Slow `bayes_search` with many observations.** Exact GP training scales cubically with the number of observations, and `interval=1` re-learns at every step.
→ Set `num_rand_basis` (e.g. 1000–5000), use `interval=5`–`10` for long runs, or reduce the candidate set.
