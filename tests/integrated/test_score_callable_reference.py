# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests: string scores reproduce the pre-callable implementation.

The reference proposals and objective values in REFERENCE were recorded
with the implementation before callable scores were introduced
(develop, commit b714f276), by running ``_run_case`` for every case in
CASES. Recording environment:

    macOS 27.0.1 (arm64), Python 3.15.1, numpy 2.5.4, scipy 1.18.1

Paths that draw random numbers through the GP posterior (TS, and every
multiple-proposal search, whose virtual observations are sampled) depend
on the BLAS/LAPACK build; for those, a mismatch is only a warning while
STRICT_STOCHASTIC_REFERENCE (conftest.py) is False, and only a sanity
check is enforced.
"""

import warnings
from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from .conftest import STRICT_STOCHASTIC_REFERENCE


MIN_X = np.array([0.0, 0.0])
MAX_X = np.array([1.0, 1.0])
# Not 24 = (1, 1) in place of 19: with 24, the map x -> (1 - x2, 1 - x1)
# swapping the two objectives of f_multi keeps the initial data, so the
# mirrored candidates 6 and 18 get mathematically equal scores, and which
# one is proposed first depends on rounding (it differed between CI runners).
INITIAL_ACTIONS = np.array([0, 7, 13, 19])
INITIAL_X_RANGE = np.array([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.5, 0.5]])
SEED = 12345
MAX_NUM_PROBES = 2
NUM_RAND_BASIS_BLM = 50
TS_CANDIDATE_NUM_REDUCED = 5


def grid_X():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


def f_single(x):
    x = np.atleast_2d(x)
    return -np.sum((x - 0.6) ** 2, axis=-1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def make_config():
    """Default search config (alpha=1.0) with 3 virtual samples and a cheap fit."""
    config = physbo.misc.SetConfig()
    config.search.multi_probe_num_sampling = 3
    config.learning.is_disp = False
    config.learning.num_init_params_search = 5
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_policy(kind):
    config = make_config()
    X = grid_X()
    if kind == "discrete":
        return physbo.search.discrete.Policy(
            test_X=X,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_single(X[INITIAL_ACTIONS])),
        )
    if kind == "range":
        return physbo.search.range.Policy(
            min_X=MIN_X,
            max_X=MAX_X,
            config=config,
            initial_data=(INITIAL_X_RANGE, f_single(INITIAL_X_RANGE)),
        )
    if kind in ("discrete_unified", "discrete_multi"):
        return getattr(physbo.search, kind).Policy(
            test_X=X,
            num_objectives=2,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_multi(X[INITIAL_ACTIONS])),
        )
    return getattr(physbo.search, kind).Policy(
        num_objectives=2,
        min_X=MIN_X,
        max_X=MAX_X,
        config=config,
        initial_data=(INITIAL_X_RANGE, f_multi(INITIAL_X_RANGE)),
    )


def make_simulator(kind):
    X = grid_X()
    if kind == "discrete":
        return lambda action: f_single(X[action])
    if kind == "range":
        return lambda x: f_single(x)
    if kind in ("discrete_unified", "discrete_multi"):
        return lambda action: f_multi(X[action])
    return lambda x: f_multi(x)


def is_range(kind):
    return kind.startswith("range")


# (kind, score, num_search_each_probe, variant)
#   variant: "gp"       GP predictor (num_rand_basis=0)
#            "blm"      BLM predictor (num_rand_basis=NUM_RAND_BASIS_BLM)
#            "reduced"  discrete_multi with TS_candidate_num=5
CASES = (
    [("discrete", s, n, "gp") for s in ["EI", "PI", "TS", "UCB"] for n in [1, 2]]
    + [("discrete", "EI", 1, "blm")]
    + [("range", s, n, "gp") for s in ["EI", "PI", "TS", "UCB"] for n in [1, 2]]
    + [("discrete_unified", s, n, "gp") for s in ["EI", "TS"] for n in [1, 2]]
    + [("range_unified", s, n, "gp") for s in ["EI", "TS"] for n in [1, 2]]
    + [("discrete_multi", s, n, "gp") for s in ["EHVI", "HVPI", "TS"] for n in [1, 2]]
    + [("discrete_multi", "TS", n, "reduced") for n in [1, 2]]
    + [("range_multi", s, n, "gp") for s in ["EHVI", "HVPI", "TS"] for n in [1, 2]]
)


def _run_case(case):
    """Run a seeded string-score search; return (proposals, fx) of the Bayes steps."""
    kind, score, N, variant = case
    policy = make_policy(kind)
    policy.set_seed(SEED)
    kwargs = {}
    if score == "UCB":
        kwargs["ucb_beta"] = 2.0
    if variant == "blm":
        kwargs["num_rand_basis"] = NUM_RAND_BASIS_BLM
    if variant == "reduced":
        policy.TS_candidate_num = TS_CANDIDATE_NUM_REDUCED
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if is_range(kind):
        kwargs["optimizer"] = physbo.search.optimize.random.Optimizer(
            min_X=MIN_X, max_X=MAX_X, nsamples=20
        )
    res = policy.bayes_search(
        max_num_probes=MAX_NUM_PROBES,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=score,
        is_disp=False,
        **kwargs,
    )
    total = res.total_num_search
    n = MAX_NUM_PROBES * N
    if is_range(kind):
        chosen = np.array(res.action_X[total - n : total])
    else:
        chosen = np.array(res.chosen_actions[total - n : total])
    fx = np.array(res.fx[total - n : total])
    return chosen, fx


def _is_stochastic(case):
    _, score, N, _ = case
    return score == "TS" or N > 1


def _case_id(case):
    return "-".join(str(c) for c in case)


# Recorded with b714f276 (see the module docstring).
# value: (proposals, fx); proposals are action indices (discrete) or X (range).
REFERENCE = {
    ('discrete', 'EI', 1, 'gp'): (
        [4, 17],
        [-0.52, -0.0325],
    ),
    ('discrete', 'EI', 2, 'gp'): (
        [4, 17, 12, 20],
        [-0.52, -0.0325, -0.01999999999999999, -0.52],
    ),
    ('discrete', 'PI', 1, 'gp'): (
        [8, 12],
        [-0.145, -0.01999999999999999],
    ),
    ('discrete', 'PI', 2, 'gp'): (
        [8, 12, 17, 20],
        [-0.145, -0.01999999999999999, -0.0325, -0.52],
    ),
    ('discrete', 'TS', 1, 'gp'): (
        [8, 16],
        [-0.145, -0.145],
    ),
    ('discrete', 'TS', 2, 'gp'): (
        [8, 9, 12, 17],
        [-0.145, -0.28250000000000003, -0.01999999999999999, -0.0325],
    ),
    ('discrete', 'UCB', 1, 'gp'): (
        [4, 16],
        [-0.52, -0.145],
    ),
    ('discrete', 'UCB', 2, 'gp'): (
        [4, 21, 17, 15],
        [-0.52, -0.28250000000000003, -0.0325, -0.3825],
    ),
    ('discrete', 'EI', 1, 'blm'): (
        [4, 17],
        [-0.52, -0.0325],
    ),
    ('range', 'EI', 1, 'gp'): (
        [[0.8622751162317551, 0.7418882012536725], [0.6931656540195079, 0.663384716182645]],
        [-0.08892049824938335, -0.012697461334437124],
    ),
    ('range', 'EI', 2, 'gp'): (
        [[0.8622751162317551, 0.7418882012536725], [0.7408265797099195, 0.6425207643476305], [0.7441691000547374, 0.9788535038184447], [0.6464394940496405, 0.627307713546643]],
        [-0.08892049824938335, -0.021640140953501044, -0.16431470676610518, -0.002902337826732113],
    ),
    ('range', 'PI', 1, 'gp'): (
        [[0.5317800493643946, 0.49159308748864405], [0.6931656540195079, 0.663384716182645]],
        [-0.016406020344969213, -0.012697461334437124],
    ),
    ('range', 'PI', 2, 'gp'): (
        [[0.5317800493643946, 0.49159308748864405], [0.7408265797099195, 0.6425207643476305], [0.7441691000547374, 0.9788535038184447], [0.6464394940496405, 0.627307713546643]],
        [-0.016406020344969213, -0.021640140953501044, -0.16431470676610518, -0.002902337826732113],
    ),
    ('range', 'TS', 1, 'gp'): (
        [[0.6931656540195079, 0.663384716182645], [0.9603662545184769, 0.4964005710409596]],
        [-0.012697461334437124, -0.14059667907631496],
    ),
    ('range', 'TS', 2, 'gp'): (
        [[0.6931656540195079, 0.663384716182645], [0.6464394940496405, 0.627307713546643], [0.7706770897406447, 0.9110390369696209], [0.4593001304067962, 0.480901504643945]],
        [-0.012697461334437124, -0.002902337826732113, -0.1258759514813253, -0.0339809048996208],
    ),
    ('range', 'UCB', 1, 'gp'): (
        [[0.8622751162317551, 0.7418882012536725], [0.628456790743723, 0.8120445187719508]],
        [-0.08892049824938335, -0.04577266688066026],
    ),
    ('range', 'UCB', 2, 'gp'): (
        [[0.8622751162317551, 0.7418882012536725], [0.628456790743723, 0.8120445187719508], [0.7441691000547374, 0.9788535038184447], [0.6464394940496405, 0.627307713546643]],
        [-0.08892049824938335, -0.04577266688066026, -0.16431470676610518, -0.002902337826732113],
    ),
    ('discrete_unified', 'EI', 1, 'gp'): (
        [8, 12],
        [[-0.25, -0.25], [-0.125, -0.125]],
    ),
    ('discrete_unified', 'EI', 2, 'gp'): (
        [8, 12, 6, 18],
        [[-0.25, -0.25], [-0.125, -0.125], [-0.0, -0.5], [-0.5, -0.0]],
    ),
    ('discrete_unified', 'TS', 1, 'gp'): (
        [8, 16],
        [[-0.25, -0.25], [-0.25, -0.25]],
    ),
    ('discrete_unified', 'TS', 2, 'gp'): (
        [8, 9, 12, 17],
        [[-0.25, -0.25], [-0.5625, -0.3125], [-0.125, -0.125], [-0.3125, -0.0625]],
    ),
    ('range_unified', 'EI', 1, 'gp'): (
        [[0.2506251745554495, 0.2722536296843763], [0.4771230301732443, 0.1454260640712819]],
        [[-0.0004956148773541345, -0.47761681063752837], [-0.06252057891070008, -0.4399714846661739]],
    ),
    ('range_unified', 'EI', 2, 'gp'): (
        [[0.2506251745554495, 0.2722536296843763], [0.2814691872358498, 0.14946697106185713], [0.19177618868080681, 0.43067291939643315], [0.2768539339658014, 0.37018041574305627]],
        [[-0.0004956148773541345, -0.47761681063752837], [-0.011097199652762442, -0.5801610413550555], [-0.03603271600776304, -0.41358360793052307], [-0.015164466097613467, -0.3681301163887558]],
    ),
    ('range_unified', 'TS', 1, 'gp'): (
        [[0.2814691872358498, 0.14946697106185713], [0.34299790623066784, 0.269140401469671]],
        [[-0.011097199652762442, -0.5801610413550555], [-0.009014965531708271, -0.39687665783136944]],
    ),
    ('range_unified', 'TS', 2, 'gp'): (
        [[0.2814691872358498, 0.14946697106185713], [0.26206897230045756, 0.28571148417546444], [0.16966411619214672, 0.3712037028160038], [0.11693077225284154, 0.4431216427222301]],
        [[-0.011097199652762442, -0.5801610413550555], [-0.0014209701944036587, -0.4536405137184817], [-0.021144191803499066, -0.48027637279534857], [-0.05500338826095781, -0.49495097328588616]],
    ),
    ('discrete_multi', 'EHVI', 1, 'gp'): (
        [4, 17],
        [[-0.625, -0.625], [-0.3125, -0.0625]],
    ),
    ('discrete_multi', 'EHVI', 2, 'gp'): (
        [4, 9, 12, 16],
        [[-0.625, -0.625], [-0.5625, -0.3125], [-0.125, -0.125], [-0.25, -0.25]],
    ),
    ('discrete_multi', 'HVPI', 1, 'gp'): (
        [8, 12],
        [[-0.25, -0.25], [-0.125, -0.125]],
    ),
    ('discrete_multi', 'HVPI', 2, 'gp'): (
        [8, 4, 12, 17],
        [[-0.25, -0.25], [-0.625, -0.625], [-0.125, -0.125], [-0.3125, -0.0625]],
    ),
    ('discrete_multi', 'TS', 1, 'gp'): (
        [9, 6],
        [[-0.5625, -0.3125], [-0.0, -0.5]],
    ),
    ('discrete_multi', 'TS', 2, 'gp'): (
        [9, 4, 1, 17],
        [[-0.5625, -0.3125], [-0.625, -0.625], [-0.0625, -0.8125], [-0.3125, -0.0625]],
    ),
    ('discrete_multi', 'TS', 1, 'reduced'): (
        [8, 21],
        [[-0.25, -0.25], [-0.5625, -0.3125]],
    ),
    ('discrete_multi', 'TS', 2, 'reduced'): (
        [8, 11, 3, 12],
        [[-0.25, -0.25], [-0.0625, -0.3125], [-0.3125, -0.5625], [-0.125, -0.125]],
    ),
    ('range_multi', 'EHVI', 1, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.24392168541812498, 0.4151960233189085]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.02732667202853759, -0.3682089632915041]],
    ),
    ('range_multi', 'EHVI', 2, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.24392168541812498, 0.4151960233189085], [0.3325047207433659, 0.2678259112621364], [0.24953997593477328, 0.32733301391499003]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.02732667202853759, -0.3682089632915041], [-0.007124792057266353, -0.40679416005176405], [-0.00598060666331663, -0.4291076168135533]],
    ),
    ('range_multi', 'HVPI', 1, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.5245804808060919, 0.41075108747616407]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.10123535256447394, -0.165903784282218]],
    ),
    ('range_multi', 'HVPI', 2, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.24392168541812498, 0.4151960233189085], [0.3325047207433659, 0.2678259112621364], [0.24953997593477328, 0.32733301391499003]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.02732667202853759, -0.3682089632915041], [-0.007124792057266353, -0.40679416005176405], [-0.00598060666331663, -0.4291076168135533]],
    ),
    ('range_multi', 'TS', 1, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.8539992203825864, 0.8447358871263595]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.7185258336587499, -0.019790726149804094]],
    ),
    ('range_multi', 'TS', 2, 'gp'): (
        [[0.36915688268789115, 0.39058979865232757], [0.7376552079735929, 0.7436941839077696], [0.6353716018471848, 0.2525650755047475], [0.4613800747509692, 0.7733130959484819]],
        [[-0.03396385417699786, -0.27421717283677915], [-0.48154154908812674, -0.0001921572067642768], [-0.14851785112261015, -0.2605811737706779], [-0.3185381323929103, -0.08384496169345924]],
    ),
}


def _sanity(case, chosen, fx):
    kind, _, N, _ = case
    n = MAX_NUM_PROBES * N
    assert len(chosen) == n
    assert np.all(np.isfinite(fx))
    if is_range(kind):
        assert chosen.shape == (n, 2)
        assert np.all(chosen >= MIN_X) and np.all(chosen <= MAX_X)
    else:
        n_cand = grid_X().shape[0]
        assert np.all((0 <= chosen) & (chosen < n_cand))
        assert len(set(chosen.tolist())) == n
        assert not set(chosen.tolist()) & set(INITIAL_ACTIONS.tolist())


def _matches(case, chosen, fx, ref_chosen, ref_fx):
    kind = case[0]
    ref_chosen = np.array(ref_chosen)
    ref_fx = np.array(ref_fx)
    if chosen.shape != ref_chosen.shape or fx.shape != ref_fx.shape:
        return False
    if is_range(kind):
        if not np.allclose(chosen, ref_chosen, rtol=0, atol=1e-8):
            return False
    elif not np.array_equal(chosen, ref_chosen):
        return False
    return np.allclose(fx, ref_fx, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_string_score_reproduces_reference(case):
    """A seeded string-score search reproduces the proposals and fx recorded before the change."""
    chosen, fx = _run_case(case)
    _sanity(case, chosen, fx)
    ref_chosen, ref_fx = REFERENCE[case]
    ok = _matches(case, chosen, fx, ref_chosen, ref_fx)
    if STRICT_STOCHASTIC_REFERENCE or not _is_stochastic(case):
        assert ok, (
            f"{_case_id(case)}: got {chosen.tolist()}, {fx.tolist()}; "
            f"expected {ref_chosen}, {ref_fx}"
        )
    elif not ok:
        warnings.warn(
            f"stochastic ({_case_id(case)}) reference mismatch: "
            f"got {chosen.tolist()}, {fx.tolist()}; "
            f"expected {ref_chosen}, {ref_fx} "
            "(environment-dependent sampling; strict check temporarily disabled)"
        )
