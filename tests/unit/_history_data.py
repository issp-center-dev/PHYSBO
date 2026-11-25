# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The observations written in test_history.py, and the history files of
tests/unit/data/history.

The files were written by running this module as a script with the PHYSBO of
the version in question on the path::

    PYTHONPATH=<checkout>/src python tests/unit/_history_data.py <directory>

It uses nothing but the History classes, so that it runs with old versions.
"""

import os
import sys

import numpy as np

SINGLE = ["discrete", "range"]
MULTI = ["discrete_multi", "range_multi", "discrete_unified", "range_unified"]
KINDS = SINGLE + MULTI

#: the number of points of each step
STEPS = [1, 3, 2, 1]
NUM_RUNS = len(STEPS)
TOTAL = sum(STEPS)
TERMINAL = np.cumsum(STEPS)

DIM = 2
NUM_OBJECTIVES = 2

ACTIONS = np.array([4, 0, 7, 2, 9, 5, 1])
X = np.array(
    [
        [0.5, 1.0],
        [0.25, -1.0],
        [2.0, 0.0],
        [-0.75, 0.125],
        [1.5, 1.5],
        [0.0, 3.0],
        [-2.0, 0.375],
    ]
)

#: single objective; the best value appears twice (the first one is kept)
FX_SINGLE = np.array([1.5, -0.5, 2.5, 2.5, 0.25, -3.0, 2.0])
#: the first observation and the one that would be the best have failed
FX_SINGLE_FAILED = np.array([np.nan, -0.5, 2.5, 2.5, 0.25, np.inf, 2.0])

#: two objectives; the front is the observations 3, 0, 2, 5
FX_MULTI = np.array(
    [
        [1.0, 5.0],
        [2.0, 2.0],
        [3.0, 4.0],
        [0.0, 6.0],
        [1.0, 1.0],
        [4.0, 1.0],
        [2.0, 3.0],
    ]
)
#: the observation 2 has failed; the front is the observations 3, 0, 6, 5
FX_MULTI_FAILED = FX_MULTI.copy()
FX_MULTI_FAILED[2, 0] = np.nan

#: the elapsed times of the steps, by the name of the argument of write()
TIMES = {
    "time_total": [10.0, 20.0, 30.0, 40.0],
    "time_update_predictor": [1.0, 2.0, 3.0, 4.0],
    "time_get_action": [0.5, 0.25, 0.125, 0.0625],
    "time_run_simulator": [8.0, 16.0, 24.0, 32.0],
}

#: the lower reference point given to the Pareto front of the files
REFERENCE_MIN = np.array([-1.0, -2.0])


def is_range(kind):
    return kind.startswith("range")


def is_multi(kind):
    return kind in MULTI


def file_name(kind):
    return kind + (".pkl" if is_multi(kind) else ".npz")


def new_history(kind):
    """An empty history of the kind, of the PHYSBO on the path."""
    import physbo

    cls = getattr(physbo.search, kind).History
    kwargs = {}
    if is_multi(kind):
        kwargs["num_objectives"] = NUM_OBJECTIVES
    if is_range(kind):
        kwargs["dim"] = DIM
    return cls(**kwargs)


def inputs(kind):
    """What the kind records of the points: their indices or the points."""
    return X if is_range(kind) else ACTIONS


def objective(kind, failed=False):
    if is_multi(kind):
        return FX_MULTI_FAILED if failed else FX_MULTI
    return FX_SINGLE_FAILED if failed else FX_SINGLE


def write_steps(history, kind, failed=False, with_times=True, steps=None):
    """Write the observations step by step, in the form each kind takes them."""
    fx = objective(kind, failed)
    points = inputs(kind)
    st = 0
    for step, N in enumerate(STEPS if steps is None else steps):
        en = st + N
        t = fx[st:en]
        if kind == "discrete":
            t = t.reshape(-1, 1)
        times = {}
        if with_times:
            times = {name: [values[step]] * N for name, values in TIMES.items()}
        history.write(t, points[st:en], **times)
        st = en
    return history


def write_files(directory):
    import physbo

    os.makedirs(directory, exist_ok=True)
    for kind in KINDS:
        history = write_steps(new_history(kind), kind)
        if is_multi(kind):
            history.pareto.set_reference_min(REFERENCE_MIN)
        # the classes of the multi-objective policies write to the name as it
        # is; numpy appends ".npz" for the others if it is not there
        history.save(os.path.join(directory, file_name(kind)))
    version = getattr(physbo, "__version__", "unknown")
    print(f"wrote {len(KINDS)} files to {directory} with PHYSBO {version}")
    print(f"  ({physbo.__file__}, numpy {np.__version__})")


if __name__ == "__main__":
    write_files(sys.argv[1])
