# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Example: When ARD (Automatic Relevance Determination) is useful

ARD is effective when the objective function depends on only a subset of
input dimensions. The GP kernel learns a separate length scale per dimension;
relevant dimensions get smaller length scales, irrelevant ones get larger
(and are effectively down-weighted).

This example uses a 6D input space where the objective depends only on
the first 2 dimensions: f(x) = -(x_0^2 + x_1^2). Dimensions 2--5 are
irrelevant. We run the same optimization with ard=True and ard=False
and compare best values, kernel length scales, and permutation importance.
"""

import numpy as np
import physbo

weights = np.array([5.0, 1.0, 0.0, 0.0])
D = len(weights)
N = 1000
np.random.seed(137)
test_X = np.random.randn(N, D)
test_X[0, :] = 0.0


def simulator(actions: np.ndarray) -> np.ndarray:
    """Objective that depends only on x_0 and x_1: f = -Σ_i w_i x_i^2."""
    X2 = test_X[actions, :] ** 2
    return -np.einsum("ai,i -> a", X2, weights)


# ---------- Run with ARD=True ----------
n_initial = 20
n_bayes = 30
n_perm = 20
score = "EI"
seed = 31415

print("=" * 60)
print("Running with ard=True")
print("=" * 60)
policy_ard = physbo.search.discrete.Policy(test_X)
policy_ard.set_seed(seed)
policy_ard.random_search(max_num_probes=n_initial, simulator=simulator, is_disp=False)
policy_ard.bayes_search(
    max_num_probes=n_bayes,
    simulator=simulator,
    score=score,
    ard=True,
    is_disp=False,
)

best_fx_ard, best_actions_ard = policy_ard.history.export_sequence_best_fx()
best_x_ard = test_X[best_actions_ard[-1], :]
print("\nBest value (ard=True):", best_fx_ard[-1])
print("Best point:", best_x_ard)

# ---------- Run with ARD=False ----------
print("\n" + "=" * 60)
print("Running with ard=False")
print("=" * 60)
policy_noard = physbo.search.discrete.Policy(test_X)
policy_noard.set_seed(seed)
policy_noard.random_search(max_num_probes=n_initial, simulator=simulator, is_disp=False)
policy_noard.bayes_search(
    max_num_probes=n_bayes,
    simulator=simulator,
    score=score,
    ard=False,
    is_disp=False,
)

best_fx_noard, best_actions_noard = policy_noard.history.export_sequence_best_fx()
best_x_noard = test_X[best_actions_noard[-1], :]
print("\nBest value (ard=False):", best_fx_noard[-1])
print("Best point:", best_x_noard)

# ---------- Compare best values ----------
print("\n" + "=" * 60)
print("Comparison: ard=True vs ard=False")
print("=" * 60)
print(f"  Best value (ard=True):  {best_fx_ard[-1]:.6f}")
print(f"  Best value (ard=False): {best_fx_noard[-1]:.6f}")
print("  (Higher is better; objective is f = -(x_0^2 + x_1^2).)")

# ---------- Compare kernel length scales ----------
print("\n--- Kernel length scale ---")
ls_ard = policy_ard.get_kernel_length_scale()
ls_noard = policy_noard.get_kernel_length_scale()
if ls_ard is not None:
    print("  ard=True:  per-dimension length scale (smaller = more relevant)")
    for i in range(len(ls_ard)):
        rel = "relevant" if i < 2 else "irrelevant"
        print(f"    dim {i} ({rel}): {ls_ard[i]:.4f}")
else:
    print("  ard=True:  (not available)")
if ls_noard is not None:
    print("  ard=False: single length scale (isotropic kernel):", ls_noard[0])
else:
    print("  ard=False: (not available)")

# ---------- Permutation importance (both) ----------
print("\n--- Permutation importance ---")
imp_mean_ard, imp_std_ard = policy_ard.get_permutation_importance(n_perm=n_perm)
imp_mean_noard, imp_std_noard = policy_noard.get_permutation_importance(n_perm=n_perm)
print("  ard=True:")
for i in range(D):
    rel = "relevant" if i < 2 else "irrelevant"
    print(f"    dim {i} ({rel}): mean = {imp_mean_ard[i]:.4f}, std = {imp_std_ard[i]:.4f}")
print("  ard=False:")
for i in range(D):
    rel = "relevant" if i < 2 else "irrelevant"
    print(f"    dim {i} ({rel}): mean = {imp_mean_noard[i]:.4f}, std = {imp_std_noard[i]:.4f}")
print("  (Higher mean => more important; both use the same GP predictor after optimization.)")
