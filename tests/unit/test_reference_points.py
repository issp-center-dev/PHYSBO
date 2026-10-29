# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The reference points of EHVI and HVPI follow the front as it grows."""

import itertools
import pickle
import warnings

import numpy as np
import numpy.testing
import pytest

from physbo.search import score_multi
from physbo.search.pareto import Pareto


def estimated_reference_points(front):
    """The documented estimates: front_min - 2 w / n and front_max + 100 w,
    where an objective without extent takes the largest width of the others
    (1 for a front without any extent)."""
    front_min = front.min(axis=0)
    front_max = front.max(axis=0)
    w = front_max - front_min
    w = np.where(w > 0, w, w.max() if np.any(w > 0) else 1.0)
    return front_min - 2 * w / front.shape[0], front_max + 100 * w


def hvi(y, front, lower, upper):
    """Hypervolume improvement of the rows of y in the box [lower, upper]
    (inclusion-exclusion over the front)."""
    out = np.zeros(len(y))
    for k in range(len(front) + 1):
        for s in itertools.combinations(range(len(front)), k):
            top = np.minimum(
                np.minimum(y, upper), np.min(front[list(s)], axis=0, initial=np.inf)
            )
            out += (-1) ** k * np.prod(np.clip(top - lower, 0, None), axis=1)
    return out


GROWN_FRONTS = [
    # a first front of a single point, then points beyond both estimates
    (
        [[1.0, 1.0]],
        [[150.0, -5.0], [-5.0, 150.0]],
        [[1.5, 1.2], [0.8, 1.6], [1.8, 0.2]],
    ),
    # a new Pareto point below the estimated reference_min (3 objectives)
    (
        [[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]],
        [[-2.0, 3.0, 1.0], [1.5, 1.5, -3.0]],
        [[1.0, 1.0, 1.0], [1.5, 1.5, 0.5], [0.0, 2.5, 1.5]],
    ),
]


@pytest.mark.parametrize("first, later, fmean", GROWN_FRONTS)
def test_ehvi_after_the_front_grows_beyond_the_estimates(first, later, fmean):
    fmean = np.array(fmean)
    fstd = np.full(fmean.shape, 0.3)
    p = Pareto(fmean.shape[1])
    p.update_front(np.array(first))
    score_multi.EHVI(fmean, fstd, p)  # estimates the reference points
    old_lower, old_upper = np.ravel(p.reference_min), np.ravel(p.reference_max)
    p.update_front(np.array(later))
    assert np.any(p.front < old_lower) or np.any(p.front > old_upper)

    ehvi = score_multi.EHVI(fmean, fstd, p)

    # an estimate the front has grown beyond is estimated again, the other
    # one is kept
    lower, upper = estimated_reference_points(p.front)
    if not np.any(p.front < old_lower):
        lower = old_lower
    if not np.any(p.front > old_upper):
        upper = old_upper
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    numpy.testing.assert_allclose(np.ravel(p.reference_max), upper)
    # Monte Carlo estimate of the expected hypervolume improvement in the box
    rng = np.random.default_rng(0)
    n = 40000
    for m, s, e in zip(fmean, fstd, ehvi):
        y = m + s * rng.standard_normal((n, len(m)))
        h = hvi(y, p.front, lower, upper)
        stderr = h.std() / np.sqrt(n)
        assert abs(e - h.mean()) < 5 * stderr + 1e-12


def test_estimates_kept_while_the_front_stays_inside():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
    fmean = np.array([[1.2, 1.2], [0.5, 1.8]])
    fstd = np.full(fmean.shape, 0.2)
    score_multi.EHVI(fmean, fstd, p)
    reference_min, reference_max = p.reference_min, p.reference_max
    # a new Pareto point between the reference points
    p.update_front(np.array([[1.5, 1.5]]))
    assert p.front_updated

    ehvi = score_multi.EHVI(fmean, fstd, p)

    assert p.reference_min is reference_min
    assert p.reference_max is reference_max
    fresh = Pareto(2)
    fresh.update_front(p.front)
    fresh.reference_min, fresh.reference_max = reference_min, reference_max
    numpy.testing.assert_array_equal(ehvi, score_multi.EHVI(fmean, fstd, fresh))


@pytest.mark.parametrize("how", ["setter", "attribute"])
def test_given_reference_points_are_kept_with_a_warning(how):
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    reference_min = np.array([[-1.0, -1.0]])
    reference_max = np.array([[3.0, 3.0]])
    if how == "setter":
        p.set_reference_min(reference_min)
        p.set_reference_max(reference_max)
    else:
        p.reference_min, p.reference_max = reference_min, reference_max
    fmean = np.array([[1.0, 1.0]])
    fstd = np.full(fmean.shape, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score_multi.EHVI(fmean, fstd, p)

    p.update_front(np.array([[-2.0, 2.5], [2.5, -2.0]]))
    with pytest.warns(RuntimeWarning, match="reference_min"):
        score_multi.EHVI(fmean, fstd, p)
    assert p.reference_min is reference_min

    p.update_front(np.array([[4.0, 0.0]]))
    with pytest.warns(RuntimeWarning, match="reference_max"):
        score_multi.EHVI(fmean, fstd, p)
    assert p.reference_max is reference_max


def test_a_given_reference_point_replaces_the_estimate():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.prepare_reference_points()
    reference_min = np.array([[-0.5, -0.5]])
    p.set_reference_min(reference_min)
    p.update_front(np.array([[-1.0, 3.0]]))
    with pytest.warns(RuntimeWarning, match="reference_min"):
        p.prepare_reference_points()
    assert p.reference_min is reference_min


def test_hvpi_needs_only_reference_min():
    p = Pareto(2)
    p.update_front(np.array([[1.0, 1.0]]))
    fmean = np.array([[1.5, 1.2], [0.8, 1.6]])
    fstd = np.full(fmean.shape, 0.3)
    score_multi.HVPI(fmean, fstd, p)
    assert p.reference_max is None
    p.update_front(np.array([[150.0, -5.0], [-5.0, 150.0]]))

    score_multi.HVPI(fmean, fstd, p)

    lower, _ = estimated_reference_points(p.front)
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    assert p.reference_max is None


def test_estimates_survive_pickling():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.prepare_reference_points()
    q = pickle.loads(pickle.dumps(p))
    q.update_front(np.array([[-3.0, 3.0]]))  # below [-2, -2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q.prepare_reference_points()
    lower, _ = estimated_reference_points(q.front)
    numpy.testing.assert_allclose(np.ravel(q.reference_min), lower)


def test_objects_of_older_versions_keep_their_reference_points():
    # objects pickled by older versions have no record of the estimates;
    # their reference points are kept as before, with a warning
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.prepare_reference_points()
    del p._estimated_reference_min
    del p._estimated_reference_max
    q = pickle.loads(pickle.dumps(p))
    reference_min = q.reference_min
    q.update_front(np.array([[-3.0, 3.0]]))  # below [-2, -2]
    with pytest.warns(RuntimeWarning, match="reference_min"):
        q.prepare_reference_points()
    numpy.testing.assert_array_equal(q.reference_min, reference_min)


@pytest.mark.parametrize("which", ["min", "max"])
def test_passing_the_estimate_makes_it_given(which):
    # set_reference_*(value) marks the value as given, even when it is the
    # estimated object itself
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.prepare_reference_points()
    estimate = getattr(p, "reference_" + which)
    getattr(p, "set_reference_" + which)(estimate)
    p.update_front(np.array([[-5.0, 2.5]]) if which == "min" else np.array([[500.0, 0.0]]))
    with pytest.warns(RuntimeWarning, match="reference_" + which):
        p.prepare_reference_points()
    assert getattr(p, "reference_" + which) is estimate


def test_warned_on_every_call_while_the_front_is_beyond():
    # no state is kept about past warnings: whether a process warns (or
    # raises) depends only on the front, the reference points and the
    # warning filters, which a parallel run shares
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.set_reference_min(np.array([[-1.0, -1.0]]))
    p.update_front(np.array([[-2.0, 3.0]]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            p.prepare_reference_points()
        assert len(caught) == 3
        p.set_reference_min(np.array([[-2.5, -2.5]]))
        p.prepare_reference_points()
        assert len(caught) == 3


@pytest.mark.parametrize("mode", ["EHVI", "HVPI"])
def test_score_prepares_the_reference_points_without_candidates(mode):
    # a process of a parallel run without candidates still updates the
    # reference points (and warns or raises like the other processes)
    from types import SimpleNamespace

    p = Pareto(2)
    p.update_front(np.array([[1.0, 1.0]]))
    p.prepare_reference_points()
    p.update_front(np.array([[150.0, -5.0], [-5.0, 150.0]]))
    empty = SimpleNamespace(X=np.zeros((0, 2)))
    out = score_multi.score(mode, None, empty, None, pareto=p)
    assert out.shape == (0,)
    lower, upper = estimated_reference_points(p.front)
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    if mode == "EHVI":
        numpy.testing.assert_allclose(np.ravel(p.reference_max), upper)
    else:
        # HVPI does not use reference_max: it is neither estimated again
        # nor warned about
        numpy.testing.assert_allclose(np.ravel(p.reference_max), [101.0, 101.0])

    q = Pareto(2)
    q.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    q.set_reference_min(np.array([[-1.0, -1.0]]))
    q.update_front(np.array([[-2.0, 3.0]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="reference_min"):
            score_multi.score(mode, None, empty, None, pareto=q)


def test_a_front_point_on_a_reference_point_is_not_beyond_it():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.set_reference_min(np.array([[0.0, 0.0]]))
    p.set_reference_max(np.array([[2.0, 2.0]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.prepare_reference_points()


def test_estimates_follow_the_front_after_loading_a_history(tmp_path):
    from physbo.search.discrete_multi import History

    h = History(num_objectives=2)
    h.write(np.array([[0.0, 2.0], [2.0, 0.0]]), np.array([0, 1]))
    h.pareto.prepare_reference_points()
    file_history = str(tmp_path / "history.pkl")
    h.save(file_history)

    h2 = History(num_objectives=2)
    h2.load(file_history)
    h2.write(np.array([[-3.0, 3.0]]), np.array([2]))  # below [-2, -2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h2.pareto.prepare_reference_points()
    lower, _ = estimated_reference_points(h2.pareto.front)
    numpy.testing.assert_allclose(np.ravel(h2.pareto.reference_min), lower)


def test_warning_raised_as_an_exception_is_raised_again():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.set_reference_min(np.array([[-1.0, -1.0]]))
    p.update_front(np.array([[-2.0, 3.0]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            with pytest.raises(RuntimeWarning, match="reference_min"):
                p.prepare_reference_points()


@pytest.mark.parametrize("mode", ["EHVI", "HVPI"])
def test_range_policy_prepares_the_reference_points_before_the_optimizer(mode):
    # in a parallel run the optimizer may evaluate the score on some
    # processes only; the reference points are prepared on all of them
    import physbo

    policy = physbo.search.range_multi.Policy(
        min_X=np.zeros(2), max_X=np.ones(2), num_objectives=2
    )
    policy.set_seed(0)
    policy.write(np.array([[0.2, 0.3]]), np.array([[1.0, 1.0]]))
    policy.history.pareto.prepare_reference_points()
    policy.write(
        np.array([[0.5, 0.1], [0.9, 0.7]]), np.array([[150.0, -5.0], [-5.0, 150.0]])
    )

    pareto = policy.history.pareto
    lower, upper = estimated_reference_points(pareto.front)
    called = []

    def never_evaluates(fn, mpicomm=None):
        # the reference points are already prepared when the optimizer runs
        numpy.testing.assert_allclose(np.ravel(pareto.reference_min), lower)
        if mode == "EHVI":
            numpy.testing.assert_allclose(np.ravel(pareto.reference_max), upper)
        else:
            # HVPI does not use reference_max
            numpy.testing.assert_allclose(np.ravel(pareto.reference_max), [101.0, 101.0])
        called.append(True)
        return np.array([0.4, 0.6])

    policy.bayes_search(
        max_num_probes=1,
        simulator=lambda x: np.array([[0.0, 0.0]]),
        score=mode,
        optimizer=never_evaluates,
        is_disp=False,
    )
    assert called


@pytest.mark.parametrize("mode", ["EHVI", "HVPI"])
@pytest.mark.parametrize("empty", [False, True])
def test_warning_location_does_not_depend_on_the_caller(mode, empty):
    # a filter on the module raises alike with and without candidates (as
    # on the processes of a parallel run) and through prepare_reference_points
    from types import SimpleNamespace

    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.set_reference_min(np.array([[-1.0, -1.0]]))
    p.update_front(np.array([[-2.0, 3.0]]))
    test = SimpleNamespace(X=np.zeros((0 if empty else 1, 2)))
    fmean = np.array([[1.0, 1.0]])
    fstd = np.full(fmean.shape, 0.2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", category=RuntimeWarning, module=r"physbo\.search\.pareto"
        )
        with pytest.raises(RuntimeWarning, match="reference_min"):
            if empty:
                score_multi.score(mode, None, test, None, pareto=p)
            else:
                getattr(score_multi, mode)(fmean, fstd, p)
        with pytest.raises(RuntimeWarning, match="reference_min"):
            p.prepare_reference_points(upper=(mode == "EHVI"))


ZERO_WIDTH_FRONTS = [
    # a single point: no extent in any objective
    ([[1.0, 1.0]], [[2.0, 2.0], [1.5, 0.5], [0.5, 1.5]], [[-1.0, -1.0]], [[101.0, 101.0]]),
    # three objectives, the front shares the third one
    (
        [[0.0, 1.0, 5.0], [1.0, 0.0, 5.0]],
        [[1.0, 1.0, 6.0], [0.5, 0.5, 5.5]],
        [[-1.0, -1.0, 4.0]],
        [[101.0, 101.0, 105.0]],
    ),
    # unequal widths of the other objectives (1 and 3): the largest is taken
    (
        [[0.0, 3.0, 5.0], [1.0, 0.0, 5.0]],
        [[1.0, 3.0, 6.0], [0.5, 1.5, 5.5]],
        [[-1.0, -3.0, 2.0]],
        [[101.0, 303.0, 305.0]],
    ),
]


@pytest.mark.parametrize("front, fmean, lower, upper", ZERO_WIDTH_FRONTS)
def test_estimates_of_a_front_without_extent(front, fmean, lower, upper):
    # an objective without extent takes the largest width of the others (1
    # when there is none), so the box between the reference points has a
    # volume and EHVI does not vanish
    fmean = np.array(fmean)
    fstd = np.full(fmean.shape, 0.2)
    p = Pareto(fmean.shape[1])
    p.update_front(np.array(front))

    ehvi = score_multi.EHVI(fmean, fstd, p)

    numpy.testing.assert_allclose(p.reference_min, lower)
    numpy.testing.assert_allclose(p.reference_max, upper)
    assert np.all(ehvi > 0)
    rng = np.random.default_rng(1)
    n = 40000
    for m, s, e in zip(fmean, fstd, ehvi):
        y = m + s * rng.standard_normal((n, len(m)))
        h = hvi(y, p.front, np.ravel(lower), np.ravel(upper))
        assert abs(e - h.mean()) < 5 * h.std() / np.sqrt(n) + 1e-12


def test_hvpi_with_a_front_of_a_single_point():
    # HVPI uses the estimated reference_min, as if it were given
    fmean = np.array([[2.0, 2.0], [1.5, 0.5], [0.5, 1.5]])
    fstd = np.full(fmean.shape, 0.2)
    p = Pareto(2)
    p.update_front(np.array([[1.0, 1.0]]))
    q = Pareto(2)
    q.update_front(np.array([[1.0, 1.0]]))
    q.set_reference_min(np.array([[-1.0, -1.0]]))

    hvpi = score_multi.HVPI(fmean, fstd, p)

    numpy.testing.assert_allclose(p.reference_min, [[-1.0, -1.0]])
    assert p.reference_max is None
    assert np.all(hvpi > 0)
    numpy.testing.assert_array_equal(hvpi, score_multi.HVPI(fmean, fstd, q))


def test_a_front_point_on_an_estimate_is_not_beyond_it():
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [2.0, 0.0]]))
    p.prepare_reference_points()
    reference_min = p.reference_min
    numpy.testing.assert_allclose(reference_min, [[-2.0, -2.0]])
    p.update_front(np.array([[-2.0, 3.0]]))  # on reference_min in the first objective
    assert p.front_updated
    p.prepare_reference_points()
    assert p.reference_min is reference_min
