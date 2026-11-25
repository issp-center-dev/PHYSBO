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

The cases above seed the legacy global state (set_seed). The "generator"
cases give rng=GENERATOR_SEED to the constructor (Generator mode) and were
recorded with develop 64c02431 in the same environment, as regression tests
for the restructuring of the policies.

With only four initial points the initial hyperparameter search often ends
with the kernel width at its lower bound; the posterior is then constant
away from the training points, every sample of the random optimizer gets
the same score, and the case pins the random stream but not the
acquisition function. GENERATOR_SEED was chosen so that this does not
happen in the generator cases, and the range cases check that the scores
of the optimizer's samples do not all tie (except where they do by
construction, see _ties_expected).
"""

import warnings
import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from .conftest import STRICT_STOCHASTIC_REFERENCE
from ._policy_fixtures import (  # noqa: E402
    MAX_X,
    MIN_X,
    f_multi,
    f_single,
    grid_X,
    is_range,
    make_config,
    make_simulator,
)

# Not 24 = (1, 1) in place of 19: with 24, the map x -> (1 - x2, 1 - x1)
# swapping the two objectives of f_multi keeps the initial data, so the
# mirrored candidates 6 and 18 get mathematically equal scores, and which
# one is proposed first depends on rounding (it differed between CI runners).
INITIAL_ACTIONS = np.array([0, 7, 13, 19])
INITIAL_X_RANGE = np.array([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.5, 0.5]])
SEED = 12345
# seed of the Generator-mode cases; with SEED their range cases degenerate
# (see the module docstring)
GENERATOR_SEED = 8
MAX_NUM_PROBES = 2
NUM_RAND_BASIS_BLM = 50
TS_CANDIDATE_NUM_REDUCED = 5


def make_policy(kind, rng=None):
    config = make_config()
    X = grid_X()
    if kind == "discrete":
        return physbo.search.discrete.Policy(
            test_X=X,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_single(X[INITIAL_ACTIONS])),
            rng=rng,
        )
    if kind == "range":
        return physbo.search.range.Policy(
            min_X=MIN_X,
            max_X=MAX_X,
            config=config,
            initial_data=(INITIAL_X_RANGE, f_single(INITIAL_X_RANGE)),
            rng=rng,
        )
    if kind in ("discrete_unified", "discrete_multi"):
        return getattr(physbo.search, kind).Policy(
            test_X=X,
            num_objectives=2,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_multi(X[INITIAL_ACTIONS])),
            rng=rng,
        )
    return getattr(physbo.search, kind).Policy(
        num_objectives=2,
        min_X=MIN_X,
        max_X=MAX_X,
        config=config,
        initial_data=(INITIAL_X_RANGE, f_multi(INITIAL_X_RANGE)),
        rng=rng,
    )


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
    # Generator mode (rng=SEED given to the constructor instead of set_seed);
    # "generator-blm" with the BLM predictor
    + [
        ("discrete", "EI", 2, "generator"),
        ("discrete", "TS", 2, "generator-blm"),
        ("range", "EI", 2, "generator"),
        ("range", "TS", 1, "generator"),
        ("discrete_multi", "EHVI", 2, "generator"),
        ("discrete_multi", "TS", 1, "generator"),
        ("discrete_unified", "TS", 2, "generator"),
        ("range_unified", "EI", 2, "generator"),
        ("range_multi", "TS", 2, "generator"),
    ]
)


class _RecordingOptimizer(physbo.search.optimize.random.Optimizer):
    """The random optimizer, recording how many distinct finite scores its
    samples got in each call (1 means that the acquisition function did not
    take part in the choice; 0 that it gave no finite score at all, in which
    case the optimizer falls back to its first sample)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distinct_scores = []

    def __call__(self, fn, mpicomm=None):
        scores = []

        def recording(x):
            s = fn(x)
            scores.append(float(np.ravel(s)[0]))
            return s

        result = super().__call__(recording, mpicomm=mpicomm)
        finite = [s for s in scores if np.isfinite(s)]
        self.distinct_scores.append(len(set(finite)))
        return result


def _run_case(case):
    """Run a seeded string-score search; return (proposals, fx, distinct) of
    the Bayes steps, where distinct lists the number of distinct scores the
    random optimizer saw in each call (empty for the discrete policies)."""
    kind, score, N, variant = case
    generator = variant.startswith("generator")
    policy = make_policy(kind, rng=GENERATOR_SEED if generator else None)
    if not generator:
        policy.set_seed(SEED)
    kwargs = {}
    if score == "UCB":
        kwargs["ucb_beta"] = 2.0
    if variant in ("blm", "generator-blm"):
        kwargs["num_rand_basis"] = NUM_RAND_BASIS_BLM
    if variant == "reduced":
        policy.TS_candidate_num = TS_CANDIDATE_NUM_REDUCED
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    optimizer = None
    if is_range(kind):
        # legacy: the global state (as before); generator: the policy's
        optimizer = _RecordingOptimizer(
            min_X=MIN_X, max_X=MAX_X, nsamples=20, rng=policy.rng if generator else None
        )
        kwargs["optimizer"] = optimizer
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
    distinct = optimizer.distinct_scores if optimizer is not None else []
    return chosen, fx, distinct


def _ties_expected(case):
    """Multi-objective TS scores one point at a time in the range policies,
    so every sample of the optimizer gets the same score (issue #144): these
    cases pin the random stream only. Strict, so that the fix shows up here."""
    kind, score, _, _ = case
    return kind == "range_multi" and score == "TS"


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
    # Generator mode, recorded with 64c02431 (see the module docstring).
    ('discrete', 'EI', 2, 'generator'): (
        [4, 17, 12, 20],
        [-0.52, -0.0325, -0.01999999999999999, -0.52],
    ),
    ('discrete', 'TS', 2, 'generator-blm'): (
        [8, 20, 12, 17],
        [-0.145, -0.52, -0.01999999999999999, -0.0325],
    ),
    ('range', 'EI', 2, 'generator'): (
        [[0.5906810288485583, 0.8725105752352116], [0.8715547262947944, 0.7098832957439117], [0.5830099222639767, 0.7191331565021924], [0.7130282441525791, 0.49890896337550616]],
        [-0.07434885683834733, -0.08581630805658466, -0.014481371719651996, -0.022994781662029798],
    ),
    ('range', 'TS', 1, 'generator'): (
        [[0.9045437744185014, 0.5326199855365639], [0.4653354003265996, 0.6197647027348296]],
        [-0.09728697688615996, -0.01852519787939337],
    ),
    ('discrete_multi', 'EHVI', 2, 'generator'): (
        [4, 21, 12, 17],
        [[-0.625, -0.625], [-0.5625, -0.3125], [-0.125, -0.125], [-0.3125, -0.0625]],
    ),
    ('discrete_multi', 'TS', 1, 'generator'): (
        [10, 3],
        [[-0.125, -0.625], [-0.3125, -0.5625]],
    ),
    ('discrete_unified', 'TS', 2, 'generator'): (
        [12, 8, 18, 6],
        [[-0.125, -0.125], [-0.25, -0.25], [-0.5, -0.0], [-0.0, -0.5]],
    ),
    ('range_unified', 'EI', 2, 'generator'): (
        [[0.2886262990454739, 0.2088484700767691], [0.9429974478333368, 0.8232222949080918], [0.38778413404848155, 0.48470246241964166], [0.47151744793960704, 0.20645475908146638]],
        [[-0.003185439392972944, -0.5057106702707299], [-0.8088292620832178, -0.04260951934178915], [-0.07406971346133323, -0.20158311699321002], [-0.05096616774832965, -0.3729939607272562]],
    ),
    ('range_multi', 'TS', 2, 'generator'): (
        [[0.38977302249907764, 0.37059760462550984], [0.16035196441985033, 0.8945732840297316], [0.5203376340912038, 0.15990252231111524], [0.6248990844438007, 0.862797536437043]],
        [[-0.034080280059938454, -0.273709652935351], [-0.4235114887682528, -0.3685862403186709], [-0.08119999189192868, -0.4009598354896097], [-0.5160701441801091, -0.028373523299265357]],
    ),
}


def _sanity(case, chosen, fx, distinct):
    kind, _, N, _ = case
    n = MAX_NUM_PROBES * N
    assert len(chosen) == n
    assert np.all(np.isfinite(fx))
    if is_range(kind):
        assert chosen.shape == (n, 2)
        assert np.all(chosen >= MIN_X) and np.all(chosen <= MAX_X)
        # the acquisition function must have taken part in every choice
        assert len(distinct) == n
        if _ties_expected(case):
            assert all(d == 1 for d in distinct), (
                f"{_case_id(case)}: distinct finite scores per step {distinct}, "
                "expected exactly one (the scores tie, issue #144); if the "
                "scores no longer tie, remove the case from _ties_expected"
            )
        else:
            assert all(d > 1 for d in distinct), (
                f"{_case_id(case)}: distinct finite scores per step {distinct}; "
                "in some step the acquisition function did not take part in "
                "the choice (all samples tied, or no score was finite)"
            )
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
    chosen, fx, distinct = _run_case(case)
    _sanity(case, chosen, fx, distinct)
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
