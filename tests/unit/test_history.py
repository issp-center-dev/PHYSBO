# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The History classes of the six search modules: what they record, export,
print, save and load.

The observations are those of _history_data.py: seven of them in steps of
1, 3, 2 and 1 points, with and without failed ones. Only the first
total_num_search entries of the arrays are looked at.
"""

import copy
import copyreg
import inspect
import os
import pickle

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search import utility  # noqa: E402
from physbo.search.pareto import Pareto  # noqa: E402

from . import _history_data as D  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history")
#: the versions that wrote the files of tests/unit/data/history
VERSIONS = ["v3.2.1", "develop_619d68fe"]

FAILED = [False, True]


def history_class(kind):
    return getattr(physbo.search, kind).History


def written(kind, failed=False, **kwargs):
    return D.write_steps(D.new_history(kind), kind, failed=failed, **kwargs)


def recorded_inputs(history, kind):
    N = history.total_num_search
    if D.is_range(kind):
        return history.action_X[0:N, :]
    return history.chosen_actions[0:N]


def best_indices(fx):
    """The index of the best valid observation after each observation: -1
    before the first valid one, and the earlier of equal values."""
    best = []
    b = -1
    for n, value in enumerate(fx):
        if np.isfinite(value) and (b < 0 or value > fx[b]):
            b = n
        best.append(b)
    return np.array(best)


def front_indices(fx):
    """The valid observations that no other one dominates (larger is
    better), in ascending order of the first objective."""
    valid = np.all(np.isfinite(fx), axis=1)

    def dominated(i):
        return any(
            valid[j] and np.all(fx[j] >= fx[i]) and np.any(fx[j] > fx[i])
            for j in range(len(fx))
        )

    front = [i for i in range(len(fx)) if valid[i] and not dominated(i)]
    return np.array(sorted(front, key=lambda i: fx[i, 0]))


def assert_record(history, kind, failed=False):
    """The counters and the observations are those of _history_data."""
    N = D.TOTAL
    assert history.num_runs == D.NUM_RUNS
    assert history.total_num_search == N
    numpy.testing.assert_array_equal(
        history.terminal_num_run[0 : D.NUM_RUNS], D.TERMINAL
    )
    numpy.testing.assert_array_equal(history.fx[0:N], D.objective(kind, failed))
    numpy.testing.assert_array_equal(recorded_inputs(history, kind), D.inputs(kind))


def assert_summary(history, kind, failed=False):
    """The best value so far, or the Pareto front, follows the observations."""
    fx = D.objective(kind, failed)
    if D.is_multi(kind):
        front, front_num = history.export_pareto_front()
        expected = front_indices(fx)
        numpy.testing.assert_array_equal(front_num, expected)
        numpy.testing.assert_array_equal(front, fx[expected])
        return

    best = best_indices(fx)
    best_fx, best_inputs = history.export_all_sequence_best_fx()
    expected_fx = np.where(best >= 0, fx[best], np.nan)
    numpy.testing.assert_array_equal(best_fx, expected_fx)
    if D.is_range(kind):
        expected_inputs = np.where((best >= 0)[:, None], D.X[best], np.nan)
    else:
        expected_inputs = np.where(best >= 0, D.ACTIONS[best], -1)
    numpy.testing.assert_array_equal(best_inputs, expected_inputs)

    # at the end of each step
    step_fx, step_inputs = history.export_sequence_best_fx()
    numpy.testing.assert_array_equal(step_fx, expected_fx[D.TERMINAL - 1])
    numpy.testing.assert_array_equal(step_inputs, expected_inputs[D.TERMINAL - 1])


# ---------------------------------------------------------------------------
# the classes


@pytest.mark.parametrize("kind", D.KINDS)
def test_each_module_has_its_own_class(kind):
    cls = history_class(kind)
    assert cls.__name__ == "History"
    assert cls.__module__ == f"physbo.search.{kind}._history"
    for other in D.KINDS:
        if other != kind:
            assert not issubclass(cls, history_class(other))


def test_constructors():
    assert physbo.search.discrete.History().total_num_search == 0
    assert physbo.search.range.History(dim=3).dim == 3
    assert physbo.search.range.History(3).dim == 3
    for kind in ["discrete_multi", "discrete_unified"]:
        assert history_class(kind)(num_objectives=3).num_objectives == 3
        assert history_class(kind)(3).num_objectives == 3
    for kind in ["range_multi", "range_unified"]:
        history = history_class(kind)(num_objectives=3, dim=4)
        assert (history.num_objectives, history.dim) == (3, 4)
        history = history_class(kind)(3, 4)
        assert (history.num_objectives, history.dim) == (3, 4)


@pytest.mark.parametrize("kind", D.KINDS)
def test_a_new_history_is_empty(kind):
    history = D.new_history(kind)
    assert history.num_runs == 0
    assert history.total_num_search == 0
    assert history.valid_mask.shape == (0,)
    points, fx = history.export_valid()
    assert len(points) == 0 and len(fx) == 0
    for name in D.TIMES:
        assert getattr(history, name).shape == (0,)
    if D.is_multi(kind):
        assert isinstance(history.pareto, Pareto)
        front, front_num = history.export_pareto_front()
        assert front.shape == (0, D.NUM_OBJECTIVES)
        assert front_num.shape == (0,)
    else:
        best_fx, best_inputs = history.export_all_sequence_best_fx()
        assert len(best_fx) == 0 and len(best_inputs) == 0
        best_fx, best_inputs = history.export_sequence_best_fx()
        assert len(best_fx) == 0 and len(best_inputs) == 0


# ---------------------------------------------------------------------------
# write and the exports


@pytest.mark.parametrize("failed", FAILED)
@pytest.mark.parametrize("kind", D.KINDS)
def test_write_records_the_observations(kind, failed):
    history = written(kind, failed)
    assert_record(history, kind, failed)

    fx = D.objective(kind, failed)
    assert history.fx.dtype == float
    if D.is_range(kind):
        assert history.action_X.dtype == float
        assert history.action_X.shape[1] == D.DIM
    else:
        assert history.chosen_actions.dtype == int
    if D.is_multi(kind):
        assert history.fx.shape[1] == D.NUM_OBJECTIVES

    valid = np.isfinite(fx) if fx.ndim == 1 else np.all(np.isfinite(fx), axis=1)
    numpy.testing.assert_array_equal(history.valid_mask, valid)
    assert history.valid_mask.dtype == bool
    points, values = history.export_valid()
    numpy.testing.assert_array_equal(points, D.inputs(kind)[valid])
    numpy.testing.assert_array_equal(values, fx[valid])


@pytest.mark.parametrize("failed", FAILED)
@pytest.mark.parametrize("kind", D.KINDS)
def test_summary_follows_the_observations(kind, failed):
    assert_summary(written(kind, failed), kind, failed)


@pytest.mark.parametrize("failed", FAILED)
def test_range_keeps_the_index_of_the_best(failed):
    history = written("range", failed)
    numpy.testing.assert_array_equal(
        history.best_index[0 : D.TOTAL], best_indices(D.objective("range", failed))
    )


@pytest.mark.parametrize("failed", FAILED)
def test_discrete_keeps_the_index_of_the_best_as_well(failed):
    history = written("discrete", failed)
    numpy.testing.assert_array_equal(
        history.best_index[0 : D.TOTAL], best_indices(D.objective("discrete", failed))
    )


@pytest.mark.parametrize("kind", D.SINGLE)
def test_arrays_of_the_times_under_their_earlier_names(kind):
    # the arrays themselves, with one entry per observation
    history = written(kind)
    for name, values in D.TIMES.items():
        numpy.testing.assert_array_equal(
            getattr(history, name + "_")[0 : D.TOTAL], np.repeat(values, D.STEPS)
        )

    # what is written to them is in the history
    history.time_total_[0] = 99.0
    assert history.time_total[0] == 99.0
    # and they can be replaced
    history.time_get_action_ = np.full(len(history.time_get_action_), 7.0)
    numpy.testing.assert_array_equal(history.time_get_action, np.full(D.NUM_RUNS, 7.0))


@pytest.mark.parametrize("kind", D.MULTI)
def test_front_after_each_step(kind):
    history = D.new_history(kind)
    updated = []
    st = 0
    for N in D.STEPS:
        en = st + N
        history.write(D.FX_MULTI[st:en], D.inputs(kind)[st:en])
        front, front_num = history.export_pareto_front()
        numpy.testing.assert_array_equal(front_num, front_indices(D.FX_MULTI[0:en]))
        updated.append(history.pareto.front_updated)
        st = en
    # the last observation is dominated
    assert updated == [True, True, True, False]


@pytest.mark.parametrize("kind", D.KINDS)
def test_elapsed_times_of_one_point_per_step(kind):
    history = D.new_history(kind)
    fx = D.objective(kind)
    points = D.inputs(kind)
    for n in range(D.NUM_RUNS):
        t = fx[n : n + 1]
        if kind == "discrete":
            t = t.reshape(-1, 1)
        times = {name: [values[n]] for name, values in D.TIMES.items()}
        history.write(t, points[n : n + 1], **times)
    for name, values in D.TIMES.items():
        numpy.testing.assert_array_equal(getattr(history, name), values)

    # a copy, not the array of the history
    history.time_total[0] = -1.0
    assert history.time_total[0] == D.TIMES["time_total"][0]


@pytest.mark.parametrize("kind", D.KINDS)
def test_elapsed_times_default_to_zero(kind):
    history = written(kind, with_times=False)
    for name in D.TIMES:
        numpy.testing.assert_array_equal(getattr(history, name), np.zeros(D.NUM_RUNS))


@pytest.mark.parametrize("kind", D.KINDS)
def test_elapsed_times_of_several_points_per_step(kind):
    # one value per step, whatever the number of its points (#156, A-1)
    history = written(kind)
    for name, values in D.TIMES.items():
        numpy.testing.assert_array_equal(getattr(history, name), values)


@pytest.mark.parametrize("kind", D.KINDS)
def test_elapsed_time_of_a_step_without_observations(kind):
    history = D.new_history(kind)
    fx = D.objective(kind)
    points = D.inputs(kind)
    as_t = (lambda t: t.reshape(-1, 1)) if kind == "discrete" else (lambda t: t)
    history.write(as_t(fx[0:2]), points[0:2], time_total=[3.0, 3.0])
    history.write(as_t(fx[0:0]), points[0:0], time_total=[])
    history.write(as_t(fx[2:3]), points[2:3], time_total=[5.0])
    history.write(as_t(fx[0:0]), points[0:0])
    assert history.num_runs == 4
    assert history.total_num_search == 3
    numpy.testing.assert_array_equal(history.time_total, [3.0, 0.0, 5.0, 0.0])


# ---------------------------------------------------------------------------
# the arguments of write


def state_of(history, kind):
    """What a write that is refused must leave as it is."""
    N = history.total_num_search
    M = history.num_runs
    state = [N, M, history.fx[0:N].copy(), recorded_inputs(history, kind).copy()]
    state.append(history.terminal_num_run[0:M].copy())
    state.extend(getattr(history, name) for name in D.TIMES)
    if D.is_multi(kind):
        state.extend(history.export_pareto_front())
    else:
        state.extend(history.export_all_sequence_best_fx())
    return state


def assert_same_state(history, kind, state):
    for now, before in zip(state_of(history, kind), state):
        numpy.testing.assert_array_equal(now, before)


@pytest.mark.parametrize("kind", D.SINGLE)
def test_single_objective_write_takes_a_vector_or_a_column(kind):
    # the discrete class took only the column, the range class only the vector
    for as_t in (lambda t: t, lambda t: t.reshape(-1, 1), lambda t: t.tolist()):
        history = D.new_history(kind)
        history.write(as_t(D.FX_SINGLE[0:3]), D.inputs(kind)[0:3])
        assert history.total_num_search == 3
        numpy.testing.assert_array_equal(history.fx[0:3], D.FX_SINGLE[0:3])
        numpy.testing.assert_array_equal(
            recorded_inputs(history, kind), D.inputs(kind)[0:3]
        )


@pytest.mark.parametrize("kind", D.SINGLE)
def test_single_objective_write_takes_one_observation(kind):
    # a value and a point, not arrays of them
    history = D.new_history(kind)
    history.write(1.5, D.inputs(kind)[0])
    history.write(np.array([2.5]), D.inputs(kind)[1].tolist())
    assert history.num_runs == 2
    assert history.total_num_search == 2
    numpy.testing.assert_array_equal(history.fx[0:2], [1.5, 2.5])
    numpy.testing.assert_array_equal(
        recorded_inputs(history, kind), D.inputs(kind)[0:2]
    )


@pytest.mark.parametrize("kind", D.SINGLE)
def test_single_objective_write_checks_the_shape_of_t(kind):
    history = written(kind, steps=D.STEPS[0:2])
    state = state_of(history, kind)
    for t in [np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((2, 1, 1))]:
        with pytest.raises(ValueError, match="t must be N values"):
            history.write(t, D.inputs(kind)[0:2])
    assert_same_state(history, kind, state)


@pytest.mark.parametrize("kind", D.KINDS)
def test_write_checks_the_shape_of_the_points(kind):
    history = written(kind, steps=D.STEPS[0:2])
    state = state_of(history, kind)
    t = D.objective(kind)[0:2]
    if D.is_range(kind):
        # points of another dimension, and too many axes
        bad = [np.zeros((2, D.DIM + 1)), np.zeros(D.DIM + 1), np.zeros((2, 1, D.DIM))]
        match = "action_X must be"
    else:
        bad = [np.zeros((2, 1), dtype=int), np.zeros((1, 2), dtype=int)]
        match = "action must be"
    for points in bad:
        with pytest.raises(ValueError, match=match):
            history.write(t, points)
    assert_same_state(history, kind, state)


@pytest.mark.parametrize("kind", D.KINDS)
def test_write_checks_the_numbers_of_values_and_points(kind):
    # the range and the multi-objective classes recorded the one point twice
    history = written(kind, steps=D.STEPS[0:2])
    state = state_of(history, kind)
    t = D.objective(kind)
    points = D.inputs(kind)
    for num_t, num_points in [(2, 1), (1, 2), (3, 2), (0, 1), (1, 0)]:
        with pytest.raises(ValueError, match=f"{num_t} != {num_points}"):
            history.write(t[0:num_t], points[0:num_points])
    assert_same_state(history, kind, state)


@pytest.mark.parametrize("kind", D.KINDS)
def test_write_checks_the_elapsed_times(kind):
    history = written(kind, steps=D.STEPS[0:2])
    state = state_of(history, kind)
    t = D.objective(kind)[0:2]
    points = D.inputs(kind)[0:2]
    for name in D.TIMES:
        for values in ([1.0, 2.0, 3.0], np.zeros((2, 1)), []):
            with pytest.raises(ValueError, match=name + " must have"):
                history.write(t, points, **{name: values})
    # the counters used to be raised before the times were looked at
    assert_same_state(history, kind, state)

    # one value is that of the step
    history.write(t, points, time_total=7.0, time_get_action=[0.5])
    assert history.time_total[-1] == 7.0
    assert history.time_get_action[-1] == 0.5
    assert history.time_run_simulator[-1] == 0.0


@pytest.mark.parametrize("kind", D.MULTI)
def test_multi_write_takes_one_point_as_a_vector(kind):
    history = D.new_history(kind)
    history.write(D.FX_MULTI[0], D.inputs(kind)[0])
    history.write(D.FX_MULTI[1:3], D.inputs(kind)[1:3])
    assert history.num_runs == 2
    assert history.total_num_search == 3
    numpy.testing.assert_array_equal(history.fx[0:3], D.FX_MULTI[0:3])
    numpy.testing.assert_array_equal(
        recorded_inputs(history, kind), D.inputs(kind)[0:3]
    )


@pytest.mark.parametrize("kind", D.MULTI)
def test_multi_write_checks_the_number_of_objectives(kind):
    history = D.new_history(kind)
    with pytest.raises(ValueError, match="number of objectives"):
        history.write(np.zeros(3), D.inputs(kind)[0])
    with pytest.raises(ValueError, match="number of objectives"):
        history.write(np.zeros((2, 3)), D.inputs(kind)[0:2])
    with pytest.raises(ValueError, match="number of objectives"):
        history.write(np.zeros((2, 1, D.NUM_OBJECTIVES)), D.inputs(kind)[0:2])
    assert history.total_num_search == 0
    assert history.num_runs == 0


# ---------------------------------------------------------------------------
# the messages

SINGLE_MESSAGES = {
    # (kind, failed, number of steps written): the message of the last step
    ("discrete", False, 1): (
        "0001-th step: f(x) = 1.500000 (action=4)\n"
        "   current best f(x) = 1.500000 (best action=4) \n\n"
    ),
    ("discrete", False, 2): (
        "current best f(x) = 2.500000 (best action=7) \n"
        "list of simulation results\n"
        "f(x)=-0.500000 (action = 0)\n"
        "f(x)=2.500000 (action = 7)\n"
        "f(x)=2.500000 (action = 2)\n"
        "\n\n"
    ),
    ("discrete", False, 4): (
        "0007-th step: f(x) = 2.000000 (action=1)\n"
        "   current best f(x) = 2.500000 (best action=7) \n\n"
    ),
    ("discrete", True, 1): (
        "0001-th step: f(x) = nan (action=4)\n"
        "   current best f(x) = (no valid observation yet) \n\n"
    ),
    ("range", False, 1): (
        "0001-th step: f(x) = 1.500000 (action=[0.5 1. ])\n"
        "   current best f(x) = 1.500000 (best action=[0.5 1. ]) \n\n"
    ),
    ("range", False, 2): (
        "current best f(x) = 2.500000 (best action=[2. 0.])\n"
        "list of simulation results\n"
        "f(x)=-0.500000 (action = [ 0.25 -1.  ])\n"
        "f(x)=2.500000 (action = [2. 0.])\n"
        "f(x)=2.500000 (action = [-0.75   0.125])\n"
        "\n\n"
    ),
    ("range", False, 4): (
        "0007-th step: f(x) = 2.000000 (action=[-2.     0.375])\n"
        "   current best f(x) = 2.500000 (best action=[2. 0.]) \n\n"
    ),
    ("range", True, 1): (
        "0001-th step: f(x) = nan (action=[0.5 1. ])\n"
        "   current best f(x) = (no valid observation yet) \n\n"
    ),
}

#: how the multi-objective classes print the points of _history_data
MULTI_POINTS = {
    "discrete": ["(action = 4)", "(action = 0)", "(action = 7)", "(action = 2)"]
    + ["(action = 9)", "(action = 5)", "(action = 1)"],
    "range": ["(X = [0.5 1. ])", "(X = [ 0.25 -1.  ])", "(X = [2. 0.])"]
    + ["(X = [-0.75   0.125])", "(X = [1.5 1.5])", "(X = [0. 3.])"]
    + ["(X = [-2.     0.375])"],
}


def multi_messages(points):
    """The messages of the multi-objective classes, by (failed, number of
    steps written, disp_pareto_set)."""
    p = points
    return {
        (False, 1, False): (
            f"0001-th step: f(x) = [1. 5.] {p[0]}\n"
            "   Pareto set updated.\n"
            "   the number of Pareto frontiers = 1 \n\n"
        ),
        (False, 1, True): (
            f"0001-th step: f(x) = [1. 5.] {p[0]}\n"
            "   Pareto set updated.\n"
            "   current Pareto set = [[1. 5.]] (steps = [1]) \n\n"
        ),
        (False, 2, False): (
            "Pareto set updated.\n"
            "the number of Pareto frontiers = 3 \n\n"
            "list of simulation results\n"
            f"f(x) = [2. 2.] {p[1]}\n"
            f"f(x) = [3. 4.] {p[2]}\n"
            f"f(x) = [0. 6.] {p[3]}\n"
            "\n\n"
        ),
        (False, 2, True): (
            "Pareto set updated.\n"
            "current Pareto set = [[0. 6.]\n [1. 5.]\n [3. 4.]] (steps = [4 1 3]) \n\n"
            "list of simulation results\n"
            f"f(x) = [2. 2.] {p[1]}\n"
            f"f(x) = [3. 4.] {p[2]}\n"
            f"f(x) = [0. 6.] {p[3]}\n"
            "\n\n"
        ),
        # the last observation does not change the front
        (False, 4, False): f"0007-th step: f(x) = [2. 3.] {p[6]}\n",
        (False, 4, True): f"0007-th step: f(x) = [2. 3.] {p[6]}\n",
        (True, 2, True): (
            "Pareto set updated.\n"
            "current Pareto set = [[0. 6.]\n [1. 5.]\n [2. 2.]] (steps = [4 1 2]) \n\n"
            "list of simulation results\n"
            f"f(x) = [2. 2.] {p[1]}\n"
            f"f(x) = [nan  4.] {p[2]}\n"
            f"f(x) = [0. 6.] {p[3]}\n"
            "\n\n"
        ),
        (True, 4, True): (
            f"0007-th step: f(x) = [2. 3.] {p[6]}\n"
            "   Pareto set updated.\n"
            "   current Pareto set = [[0. 6.]\n [1. 5.]\n [2. 3.]\n [4. 1.]]"
            " (steps = [4 1 7 6]) \n\n"
        ),
    }


@pytest.mark.parametrize("case", sorted(SINGLE_MESSAGES))
def test_messages_of_the_single_objective_classes(case, capsys):
    kind, failed, num_steps = case
    history = written(kind, failed, steps=D.STEPS[0:num_steps])
    capsys.readouterr()
    history.show_search_results(D.STEPS[num_steps - 1])
    assert capsys.readouterr().out == SINGLE_MESSAGES[case]


@pytest.mark.parametrize("case", sorted(multi_messages(MULTI_POINTS["discrete"])))
@pytest.mark.parametrize("kind", D.MULTI)
def test_messages_of_the_multi_objective_classes(kind, case, capsys):
    failed, num_steps, disp_pareto_set = case
    points = MULTI_POINTS["range" if D.is_range(kind) else "discrete"]
    history = written(kind, failed, steps=D.STEPS[0:num_steps])
    capsys.readouterr()
    history.show_search_results_mo(
        D.STEPS[num_steps - 1], disp_pareto_set=disp_pareto_set
    )
    assert capsys.readouterr().out == multi_messages(points)[case]


@pytest.mark.parametrize("kind", D.KINDS)
def test_functions_of_utility_print_the_same(kind, capsys):
    history = written(kind)
    if D.is_multi(kind):
        history.show_search_results_mo(1, disp_pareto_set=True)
        expected = capsys.readouterr().out
        utility.show_search_results_mo(history, 1, disp_pareto_set=True)
    else:
        history.show_search_results(1)
        expected = capsys.readouterr().out
        utility.show_search_results(history, 1)
    assert expected != ""
    assert capsys.readouterr().out == expected


# ---------------------------------------------------------------------------
# save and load


def assert_times_after_load(history, kind):
    """Only the single-objective range class saves the times (#156, A-4)."""
    for name, values in D.TIMES.items():
        if kind == "range":
            expected = values
        else:
            expected = np.zeros(D.NUM_RUNS)
        numpy.testing.assert_array_equal(getattr(history, name), expected)


@pytest.mark.parametrize("failed", FAILED)
@pytest.mark.parametrize("kind", D.KINDS)
def test_save_and_load(kind, failed, tmp_path):
    history = written(kind, failed)
    if D.is_multi(kind):
        history.pareto.set_reference_min(D.REFERENCE_MIN)
    file_name = str(tmp_path / D.file_name(kind))
    history.save(file_name)
    assert os.listdir(tmp_path) == [D.file_name(kind)]

    loaded = D.new_history(kind)
    loaded.load(file_name)
    assert_record(loaded, kind, failed)
    assert_summary(loaded, kind, failed)
    assert_times_after_load(loaded, kind)
    if D.is_multi(kind):
        assert isinstance(loaded.pareto, Pareto)
        numpy.testing.assert_array_equal(loaded.pareto.reference_min, D.REFERENCE_MIN)
        assert loaded.pareto.reference_max is None
    if kind == "range":
        numpy.testing.assert_array_equal(
            loaded.best_index[0 : D.TOTAL], best_indices(D.objective(kind, failed))
        )


@pytest.mark.parametrize("kind", D.SINGLE)
def test_save_appends_the_suffix_of_numpy(kind, tmp_path):
    written(kind).save(str(tmp_path / "history"))
    assert os.listdir(tmp_path) == ["history.npz"]
    loaded = D.new_history(kind)
    loaded.load(str(tmp_path / "history.npz"))
    assert_record(loaded, kind)


@pytest.mark.parametrize("kind", D.KINDS)
def test_a_loaded_history_goes_on(kind, tmp_path):
    history = written(kind, steps=D.STEPS[0:2])
    file_name = str(tmp_path / D.file_name(kind))
    history.save(file_name)
    loaded = D.new_history(kind)
    loaded.load(file_name)

    # the remaining steps
    st = sum(D.STEPS[0:2])
    for N in D.STEPS[2:]:
        en = st + N
        t = D.objective(kind)[st:en]
        if kind == "discrete":
            t = t.reshape(-1, 1)
        loaded.write(t, D.inputs(kind)[st:en])
        st = en
    assert_record(loaded, kind)
    assert_summary(loaded, kind)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kind", D.KINDS)
def test_files_of_earlier_versions(kind, version):
    if D.is_multi(kind):
        # the files hold arrays pickled by numpy 2
        pytest.importorskip("numpy._core")
    history = D.new_history(kind)
    history.load(os.path.join(DATA, version, D.file_name(kind)))
    assert_record(history, kind)
    assert_summary(history, kind)
    assert_times_after_load(history, kind)
    if D.is_multi(kind):
        numpy.testing.assert_array_equal(history.pareto.reference_min, D.REFERENCE_MIN)
        assert history.pareto.reference_max is None

    # and it goes on: a point that is better than all
    best = np.full((1, D.NUM_OBJECTIVES), 10.0)
    if D.is_multi(kind):
        history.write(best, D.inputs(kind)[0:1])
        front, front_num = history.export_pareto_front()
        numpy.testing.assert_array_equal(front, best)
        numpy.testing.assert_array_equal(front_num, [D.TOTAL])
    else:
        t = best[:, 0]
        history.write(
            t.reshape(-1, 1) if kind == "discrete" else t, D.inputs(kind)[0:1]
        )
        best_fx, _ = history.export_all_sequence_best_fx()
        assert best_fx[-1] == 10.0
    assert history.total_num_search == D.TOTAL + 1
    assert history.num_runs == D.NUM_RUNS + 1


# ---------------------------------------------------------------------------
# copies


@pytest.mark.parametrize(
    "duplicate", [copy.deepcopy, lambda h: pickle.loads(pickle.dumps(h))]
)
@pytest.mark.parametrize("kind", D.KINDS)
def test_copies_are_independent(kind, duplicate):
    history = written(kind, steps=D.STEPS[0:3])
    clone = duplicate(history)
    assert type(clone) is type(history)

    # the last step, to the copy only
    st = sum(D.STEPS[0:3])
    t = D.objective(kind)[st:]
    if kind == "discrete":
        t = t.reshape(-1, 1)
    times = {name: [values[-1]] for name, values in D.TIMES.items()}
    clone.write(t, D.inputs(kind)[st:], **times)
    assert_record(clone, kind)
    assert_summary(clone, kind)
    assert history.total_num_search == st
    assert history.num_runs == 3
    if D.is_multi(kind):
        _, front_num = history.export_pareto_front()
        numpy.testing.assert_array_equal(front_num, front_indices(D.FX_MULTI[0:st]))


# ---------------------------------------------------------------------------
# what a change of the classes must keep


@pytest.mark.parametrize("kind", D.KINDS)
def test_signature_of_write(kind):
    names = list(inspect.signature(history_class(kind).write).parameters)
    points = "action_X" if D.is_range(kind) else "action"
    assert names == ["self", "t", points] + list(D.TIMES)


@pytest.mark.parametrize("kind", D.KINDS)
def test_entries_of_the_saved_file(kind, tmp_path):
    file_name = str(tmp_path / D.file_name(kind))
    written(kind).save(file_name)

    expected = {"num_runs", "total_num_search", "fx", "terminal_num_run"}
    expected.add("action_X" if D.is_range(kind) else "chosen_actions")
    if D.is_multi(kind):
        expected.add("pareto")
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data["pareto"], Pareto)
    else:
        if kind == "range":
            expected |= {"best_index"} | set(D.TIMES)
        data = dict(np.load(file_name))
    assert set(data) == expected
    assert int(data["num_runs"]) == D.NUM_RUNS
    assert int(data["total_num_search"]) == D.TOTAL
    numpy.testing.assert_array_equal(data["fx"], D.objective(kind))
    numpy.testing.assert_array_equal(data["terminal_num_run"], D.TERMINAL)
    if kind == "range":
        # the times, per observation
        for name, values in D.TIMES.items():
            numpy.testing.assert_array_equal(data[name], np.repeat(values, D.STEPS))


@pytest.mark.parametrize("kind", D.SINGLE)
def test_message_when_the_last_observation_is_the_best(kind, capsys):
    history = D.new_history(kind)
    for n, value in enumerate([1.0, 3.0]):
        t = np.array([value])
        history.write(
            t.reshape(-1, 1) if kind == "discrete" else t, D.inputs(kind)[n : n + 1]
        )
    capsys.readouterr()
    history.show_search_results(1)
    point = "[ 0.25 -1.  ]" if kind == "range" else "0"
    assert capsys.readouterr().out == (
        f"0002-th step: f(x) = 3.000000 (action={point})\n"
        f"   current best f(x) = 3.000000 (best action={point}) \n\n"
    )


#: the number of observations a history has room for (#156, A-2)
ROOM = 30000


def full_history(kind):
    history = D.new_history(kind)
    t = np.arange(ROOM, dtype=float)
    points = np.zeros((ROOM, D.DIM)) if D.is_range(kind) else np.arange(ROOM)
    history.write(t.reshape(-1, 1) if kind == "discrete" else t, points)
    assert history.total_num_search == ROOM
    best_fx, _ = history.export_all_sequence_best_fx()
    assert best_fx[-1] == ROOM - 1
    return history


def write_one_more(history, kind):
    t = np.array([-1.0])
    history.write(t.reshape(-1, 1) if kind == "discrete" else t, D.inputs(kind)[0:1])


def test_a_full_range_history_refuses_more():
    history = full_history("range")
    with pytest.raises(IndexError):
        write_one_more(history, "range")
    assert history.total_num_search == ROOM
    assert history.num_runs == 1


def test_a_full_discrete_history_refuses_more_as_well():
    # it used to drop the observation and count it
    history = full_history("discrete")
    with pytest.raises(IndexError):
        write_one_more(history, "discrete")
    assert history.total_num_search == ROOM
    assert history.num_runs == 1


class PickledEarlier:
    """Pickles as a single-objective history did before the classes shared
    their base: the arrays of the elapsed times were time_total_ and so on,
    and the discrete class had no best_index."""

    def __init__(self, history):
        self.cls = type(history)
        self.state = dict(vars(history))
        for name in D.TIMES:
            self.state[name + "_"] = self.state.pop("_" + name)
        if "chosen_actions" in self.state:
            del self.state["best_index"]

    def __reduce__(self):
        # an instance made without __init__, then given the state
        return (copyreg._reconstructor, (self.cls, object, None), self.state)


@pytest.mark.parametrize("failed", FAILED)
@pytest.mark.parametrize("kind", D.SINGLE)
def test_history_pickled_before_the_common_base(kind, failed):
    earlier = pickle.loads(pickle.dumps(PickledEarlier(written(kind, failed))))
    assert type(earlier) is history_class(kind)
    assert_record(earlier, kind, failed)
    assert_summary(earlier, kind, failed)
    numpy.testing.assert_array_equal(
        earlier.best_index[0 : D.TOTAL], best_indices(D.objective(kind, failed))
    )
    for name, values in D.TIMES.items():
        numpy.testing.assert_array_equal(getattr(earlier, name), values)

    # and it goes on
    write_one_more(earlier, kind)
    assert earlier.total_num_search == D.TOTAL + 1
    best_fx, _ = earlier.export_all_sequence_best_fx()
    assert best_fx[-1] == 2.5


# ---------------------------------------------------------------------------
# the policies


def make_policy(kind):
    module = getattr(physbo.search, kind)
    kwargs = {"rng": np.random.default_rng(0)}
    if D.is_range(kind):
        kwargs.update(min_X=[-2.0] * D.DIM, max_X=[2.0] * D.DIM)
    else:
        kwargs["test_X"] = np.random.default_rng(1).random((20, D.DIM))
    if D.is_multi(kind):
        kwargs["num_objectives"] = D.NUM_OBJECTIVES
    return module.Policy(**kwargs)


@pytest.mark.parametrize("kind", D.KINDS)
def test_policy_writes_to_the_history_of_its_module(kind):
    policy = make_policy(kind)
    assert type(policy.history) is history_class(kind)

    def simulator(action):
        x = action if D.is_range(kind) else policy.test.X[action]
        f = -np.sum(x**2, axis=1)
        return np.c_[f, -np.sum((x - 1) ** 2, axis=1)] if D.is_multi(kind) else f

    result = policy.random_search(
        max_num_probes=2, num_search_each_probe=2, simulator=simulator, is_disp=False
    )
    history = policy.history
    assert type(result) is history_class(kind)
    assert result is not history
    assert policy.export_history() is history
    assert history.num_runs == 2
    assert history.total_num_search == 4
    numpy.testing.assert_array_equal(history.terminal_num_run[0:2], [2, 4])
    numpy.testing.assert_array_equal(result.fx[0:4], history.fx[0:4])
    numpy.testing.assert_array_equal(
        recorded_inputs(result, kind), recorded_inputs(history, kind)
    )
    points = recorded_inputs(history, kind)
    numpy.testing.assert_array_equal(history.fx[0:4], simulator(points))


@pytest.mark.parametrize("kind", D.KINDS)
def test_policy_records_the_times_of_the_steps(kind):
    policy = make_policy(kind)

    def simulator(action):
        x = action if D.is_range(kind) else policy.test.X[action]
        f = -np.sum(x**2, axis=1)
        return np.c_[f, -np.sum((x - 1) ** 2, axis=1)] if D.is_multi(kind) else f

    policy.random_search(
        max_num_probes=3, num_search_each_probe=4, simulator=simulator, is_disp=False
    )
    history = policy.history
    for name in D.TIMES:
        times = getattr(history, name)
        assert times.shape == (3,)
        assert np.all(times >= 0.0)
    assert np.all(history.time_total > 0.0)
    assert np.all(history.time_total >= history.time_run_simulator)
    assert np.all(history.time_total >= history.time_get_action)
