# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The history of a search, for the six search modules.

A history records the observations of a search in the order they were made:
the points, the values of the objective(s), the step each belongs to and the
elapsed times. It also keeps a summary of them up to date, which is the best
value so far for a single objective and the Pareto front for several.

The six ``History`` classes differ in two ways only:

- the search space decides how a point is recorded: by its index among the
  candidates (``chosen_actions``) or by its coordinates (``action_X``).
  This is described by :class:`_ActionColumn` and :class:`_PointColumn`;
- the objective decides the shape of ``fx`` and the summary:
  :class:`_SingleObjectiveHistory` and :class:`_MultiObjectiveHistory`.

Everything else is in :class:`_HistoryBase`. The classes of the search
modules derive from the four combinations at the end of this module.
"""

import copy
import pickle

import numpy as np

from . import pareto
from . import utility

#: the number of observations a history has room for
MAX_SEARCH = int(30000)

#: the elapsed times recorded with each observation, and what they measure
_TIMES = {
    "time_total": "the whole step",
    "time_update_predictor": "updating the predictor",
    "time_get_action": "choosing the next points",
    "time_run_simulator": "running the simulator",
}


def _elapsed_time(name):
    """The property that returns a copy of the times of the steps."""

    def getter(self):
        return copy.copy(getattr(self, "_" + name)[0 : self.num_runs])

    return property(getter, doc=f"The elapsed time of {_TIMES[name]}, for each step.")


def _time_array(name):
    """The property that gives the array of an elapsed time itself, under the
    name it had as an attribute of the single-objective classes."""

    def getter(self):
        return getattr(self, "_" + name)

    def setter(self, values):
        setattr(self, "_" + name, values)

    doc = (
        f"The array behind ``{name}``, with one entry per observation"
        " (kept for compatibility)."
    )
    return property(getter, setter, doc=doc)


class _HistoryBase:
    """What the histories of all search modules have in common.

    A class of a search module combines a class of this kind with a
    description of the column of the points (:class:`_ActionColumn` or
    :class:`_PointColumn`), and calls :meth:`_allocate` in its constructor.
    """

    def _allocate(self):
        """Make the arrays of an empty history."""
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros((MAX_SEARCH,) + self._fx_shape(), dtype=float)
        setattr(
            self,
            self._column_name,
            np.zeros((MAX_SEARCH,) + self._column_shape(), dtype=self._column_dtype),
        )
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)
        for name in _TIMES:
            setattr(self, "_" + name, np.zeros(MAX_SEARCH, dtype=float))

    def _fx_shape(self):
        """The shape of the objective value(s) of one observation."""
        raise NotImplementedError

    @property
    def _points(self):
        """The array that records the points (chosen_actions or action_X)."""
        return getattr(self, self._column_name)

    time_total = _elapsed_time("time_total")
    time_update_predictor = _elapsed_time("time_update_predictor")
    time_get_action = _elapsed_time("time_get_action")
    time_run_simulator = _elapsed_time("time_run_simulator")

    @property
    def valid_mask(self):
        """
        Mask of valid observations (True) vs failed ones (False).

        An observation is failed when its objective value (any of them, for
        several objectives) is not finite (NaN or +-Inf). Failed observations
        are kept in the history but excluded from the training data and from
        the best value so far or the Pareto front.

        Returns
        -------
        numpy.ndarray of bool, shape (total_num_search,)
        """
        return utility.finite_mask(self.fx[0 : self.total_num_search])

    def export_valid(self):
        """
        Export the valid (successfully evaluated) observations.

        Returns
        -------
        points: numpy.ndarray
            The points of the valid observations: the indexes of the actions
            in a discrete search space, the inputs (N_valid x dim) in a
            continuous one.
        fx: numpy.ndarray
            Objective values of the valid observations (N_valid x
            num_objectives for several objectives).
        """
        N = self.total_num_search
        mask = self.valid_mask
        return self._points[0:N][mask], self.fx[0:N][mask]

    def _write(
        self,
        t,
        points,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        """Record the observations of one step (the body of ``write``)."""
        t, points, N = self._check_observations(t, points)
        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en] = t
        self._points[st:en] = points
        self._update_summary(st, en, t)

        self.num_runs += 1
        self.total_num_search += N

        times = (time_total, time_update_predictor, time_get_action, time_run_simulator)
        for name, values in zip(_TIMES, times):
            if values is None:
                values = np.zeros(N, dtype=float)
            getattr(self, "_" + name)[st:en] = values

    def _check_observations(self, t, points):
        """Check the arguments of ``write``.

        Returns
        -------
        The objective values and the points as they are assigned to the
        arrays, and the number of observations.
        """
        raise NotImplementedError

    def _update_summary(self, st, en, t):
        """Bring the summary up to date with the observations st to en."""
        raise NotImplementedError

    def _saved_arrays(self):
        """The entries of the file of ``save`` that all classes write."""
        N = self.total_num_search
        M = self.num_runs
        return {
            "num_runs": M,
            "total_num_search": N,
            "fx": self.fx[0:N],
            self._column_name: self._points[0:N],
            "terminal_num_run": self.terminal_num_run[0:M],
        }

    def _restore_arrays(self, data):
        """Take the entries of :meth:`_saved_arrays` from a loaded file."""
        M = int(data["num_runs"])
        N = int(data["total_num_search"])
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data["fx"]
        self._points[0:N] = data[self._column_name]
        self.terminal_num_run[0:M] = data["terminal_num_run"]


# ---------------------------------------------------------------------------
# the search space: how a point is recorded


class _ActionColumn:
    """The points are recorded by their index among the candidates
    (discrete search space)."""

    _column_name = "chosen_actions"
    _column_dtype = int
    #: stands for the best point while there is no valid observation
    _column_missing = -1
    #: how the multi-objective messages call a point
    _column_label = "action"

    def _column_shape(self):
        return ()

    @staticmethod
    def _format_point(point):
        return "%d" % point

    def write(
        self,
        t,
        action,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        """
        Record the observations of one step.

        Parameters
        ----------
        t: numpy.ndarray
            The values of the objective function(s) of the observations
            (to be maximized). For a single objective, an N x 1 array; for
            several objectives, an N x num_objectives array, or an array of
            num_objectives values for one observation.
        action: numpy.ndarray
            N dimensional array. The indexes of the actions of the
            observations.
        time_total: numpy.ndarray
            N dimensional array. The total elapsed time of the step, for
            each observation. If None (default), filled by 0.0.
        time_update_predictor: numpy.ndarray
            N dimensional array. The elapsed time for updating the predictor
            (e.g., learning the hyperparameters). If None (default), filled
            by 0.0.
        time_get_action: numpy.ndarray
            N dimensional array. The elapsed time for getting the next
            action. If None (default), filled by 0.0.
        time_run_simulator: numpy.ndarray
            N dimensional array. The elapsed time for running the simulator.
            If None (default), filled by 0.0.
        """
        self._write(
            t,
            action,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )


class _PointColumn:
    """The points are recorded by their coordinates (continuous search
    space). The history has the dimension of the space as ``dim``."""

    _column_name = "action_X"
    _column_dtype = float
    _column_missing = np.nan
    _column_label = "X"

    def _column_shape(self):
        return (self.dim,)

    @staticmethod
    def _format_point(point):
        return str(point)

    def write(
        self,
        t,
        action_X,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        """
        Record the observations of one step.

        Parameters
        ----------
        t: numpy.ndarray
            The values of the objective function(s) of the observations
            (to be maximized). For a single objective, an N dimensional
            array; for several objectives, an N x num_objectives array, or
            an array of num_objectives values for one observation.
        action_X: numpy.ndarray
            N x d dimensional array. The inputs of the observations.
        time_total: numpy.ndarray
            N dimensional array. The total elapsed time of the step, for
            each observation. If None (default), filled by 0.0.
        time_update_predictor: numpy.ndarray
            N dimensional array. The elapsed time for updating the predictor
            (e.g., learning the hyperparameters). If None (default), filled
            by 0.0.
        time_get_action: numpy.ndarray
            N dimensional array. The elapsed time for getting the next
            action. If None (default), filled by 0.0.
        time_run_simulator: numpy.ndarray
            N dimensional array. The elapsed time for running the simulator.
            If None (default), filled by 0.0.
        """
        self._write(
            t,
            action_X,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )


# ---------------------------------------------------------------------------
# the objective: the shape of fx and the summary


class _SingleObjectiveHistory(_HistoryBase):
    """The history of a single objective. Its summary is the best valid
    observation so far, kept as ``best_index``: the index of that observation
    after each observation, -1 until the first valid one."""

    #: what follows the best value in the message of a step of several
    #: points (the two search spaces have differed in this)
    _best_message_end = ""

    def _allocate(self):
        super()._allocate()
        self.best_index = np.full(MAX_SEARCH, -1, dtype=int)

    def _fx_shape(self):
        return ()

    # the arrays of the elapsed times under their earlier names
    time_total_ = _time_array("time_total")
    time_update_predictor_ = _time_array("time_update_predictor")
    time_get_action_ = _time_array("time_get_action")
    time_run_simulator_ = _time_array("time_run_simulator")

    def __setstate__(self, state):
        # A history pickled before the classes shared their base has the
        # arrays of the elapsed times under their earlier names and, for a
        # discrete search space, no best_index.
        state = dict(state)
        for name in _TIMES:
            if name + "_" in state:
                state["_" + name] = state.pop(name + "_")
        self.__dict__.update(state)
        if "best_index" not in state:
            self.best_index = np.full(len(self.fx), -1, dtype=int)
            self._update_best_index(0, self.total_num_search)

    def _update_summary(self, st, en, t):
        self._update_best_index(st, en)

    def _update_best_index(self, st, en):
        """
        Update best_index[st:en] from fx.

        Failed (non-finite) observations are skipped, and best_index stays -1
        until the first valid one. Of equal values the earlier one is kept.
        """
        for n in range(st, en):
            prev = self.best_index[n - 1] if n > 0 else -1
            if np.isfinite(self.fx[n]) and (prev < 0 or self.fx[n] > self.fx[prev]):
                self.best_index[n] = n
            else:
                self.best_index[n] = prev

    def _best(self, index):
        """The values and the points of the observations ``index``, with NaN
        and the missing point where the index is -1."""
        index = np.asarray(index, dtype=int)
        found = index >= 0
        best_fx = np.full(len(index), np.nan, dtype=float)
        best_fx[found] = self.fx[index[found]]
        best_points = np.full(
            (len(index),) + self._column_shape(),
            self._column_missing,
            dtype=self._column_dtype,
        )
        best_points[found] = self._points[index[found]]
        return best_fx, best_points

    def export_sequence_best_fx(self):
        """
        Export the best fx and its point at the end of each step (each call
        of write).

        Until the first valid observation, the best fx is NaN and the point
        is -1 (the index of an action) or NaN (the coordinates of a point).

        Returns
        -------
        best_fx: numpy.ndarray (num_runs)
            The best fx at the end of each step.
        best_points: numpy.ndarray
            The index of its action (num_runs) in a discrete search space,
            its input (num_runs, dim) in a continuous one.
        """
        last = self.terminal_num_run[0 : self.num_runs] - 1
        return self._best(self.best_index[last])

    def export_all_sequence_best_fx(self):
        """
        Export the best fx and its point after each observation.
        (The total number of data is total_num_search.)

        Until the first valid observation, the best fx is NaN and the point
        is -1 (the index of an action) or NaN (the coordinates of a point).

        Returns
        -------
        best_fx: numpy.ndarray (total_num_search)
        best_points: numpy.ndarray
            The indexes of the actions (total_num_search) in a discrete
            search space, the inputs (total_num_search, dim) in a continuous
            one.
        """
        return self._best(self.best_index[0 : self.total_num_search])

    def save(self, filename):
        """
        Save the information of the history.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
            (numpy appends ".npz" if the name does not end with it)
        """
        np.savez_compressed(filename, **self._saved_arrays())

    def load(self, filename):
        """
        Load the information of the history.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
        """
        data = np.load(filename)
        self._restore_arrays(data)
        # best_index is recomputed from fx: the saved one may refer to a
        # failed observation if the file was written by an older version
        self._update_best_index(0, self.total_num_search)
        self._restore_more(data)

    def _restore_more(self, data):
        """Take what the class saves in addition from a loaded file."""

    def show_search_results(self, N):
        """Print the last N observations and the best value so far."""
        n = self.total_num_search
        fmt = self._format_point
        index = self.best_index[n - 1]
        if index >= 0:
            best_msg = "current best f(x) = %f (best action=%s)" % (
                self.fx[index],
                fmt(self._points[index]),
            )
        else:
            best_msg = "current best f(x) = (no valid observation yet)"

        if N == 1:
            print(
                "%04d-th step: f(x) = %f (action=%s)"
                % (n, self.fx[n - 1], fmt(self._points[n - 1]))
            )
            print("   " + best_msg + " \n")
        else:
            print(best_msg + self._best_message_end)

            print("list of simulation results")
            st = self.total_num_search - N
            en = self.total_num_search
            for n in range(st, en):
                print("f(x)=%f (action = %s)" % (self.fx[n], fmt(self._points[n])))
            print("\n")


class _MultiObjectiveHistory(_HistoryBase):
    """The history of several objectives (``num_objectives``). Its summary is
    the Pareto front of the valid observations, ``pareto``."""

    def _allocate(self):
        self.pareto = pareto.Pareto(num_objectives=self.num_objectives)
        super()._allocate()

    def _fx_shape(self):
        return (self.num_objectives,)

    def _check_observations(self, t, points):
        t = np.array(t)
        points = np.array(points)

        if t.ndim == 1:
            N = 1
            if len(t) != self.num_objectives:
                raise ValueError("t does not match the number of objectives")
        else:
            N = t.shape[0]
            if t.shape[1] != self.num_objectives:
                raise ValueError("t does not match the number of objectives")
        return t, points, N

    def _update_summary(self, st, en, t):
        self.pareto.update_front(t)

    def export_pareto_front(self):
        """
        Export the Pareto front of the valid observations.

        Returns
        -------
        front: numpy.ndarray
            The objective values of the non-dominated observations, in
            ascending order of the first objective.
        front_num: numpy.ndarray
            The indexes of these observations in the history.
        """
        return self.pareto.export_front()

    def save(self, filename):
        """
        Save the information of the history, with the Pareto front.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
        """
        obj = self._saved_arrays()
        obj["pareto"] = self.pareto

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def load(self, filename):
        """
        Load the information of the history.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self._restore_arrays(data)
        self.pareto = data["pareto"]
        # the front is recomputed from fx: the saved one may contain failed
        # observations if the file was written by an older version
        self.pareto.rebuild_front(
            self.fx[0 : self.total_num_search],
            self.terminal_num_run[0 : self.num_runs],
        )

    def show_search_results_mo(self, N, disp_pareto_set=False):
        """Print the last N observations, and the Pareto front if they have
        changed it (the front itself with ``disp_pareto_set``, else its
        size)."""
        n = self.total_num_search
        pset, step = self.pareto.export_front()

        def point(i):
            return "(%s = %s)" % (
                self._column_label,
                self._format_point(self._points[i]),
            )

        def msg_pareto_set_updated(indent=False):
            prefix = "   " if indent else ""
            if self.pareto.front_updated:
                print(prefix + "Pareto set updated.")
                if disp_pareto_set:
                    print(
                        prefix
                        + "current Pareto set = %s (steps = %s) \n"
                        % (str(pset), str(step + 1))
                    )
                else:
                    print(
                        prefix
                        + "the number of Pareto frontiers = %s \n" % str(len(step))
                    )

        if N == 1:
            print("%04d-th step: f(x) = %s %s" % (n, str(self.fx[n - 1]), point(n - 1)))

            msg_pareto_set_updated(indent=True)

        else:
            msg_pareto_set_updated()

            print("list of simulation results")
            st = self.total_num_search - N
            en = self.total_num_search
            for n in range(st, en):
                print("f(x) = %s %s" % (str(self.fx[n]), point(n)))
            print("\n")


# ---------------------------------------------------------------------------
# the four combinations


class _DiscreteHistory(_ActionColumn, _SingleObjectiveHistory):
    """Single objective, discrete search space."""

    _best_message_end = " "

    def __init__(self):
        self._allocate()

    def _check_observations(self, t, points):
        N = utility.length_vector(points)
        t_shape = t.shape
        if t_shape[0] != N:
            raise ValueError(
                f"Number of actions and t must be the same: {N} != {t_shape[0]}"
            )
        if t_shape[1] != 1:
            raise ValueError(f"t.shape[1] must be 1: {t_shape}")
        return t[:, 0], points, N


class _RangeHistory(_PointColumn, _SingleObjectiveHistory):
    """Single objective, continuous search space. Unlike the other classes,
    it saves ``best_index`` and the elapsed times."""

    def __init__(self, dim: int):
        self.dim = dim
        self._allocate()

    def _check_observations(self, t, points):
        return t, points, utility.length_vector(t)

    def _saved_arrays(self):
        N = self.total_num_search
        data = super()._saved_arrays()
        # in the order of the entries of the files written so far
        terminal_num_run = data.pop("terminal_num_run")
        data["best_index"] = self.best_index[0:N]
        data["terminal_num_run"] = terminal_num_run
        for name in _TIMES:
            data[name] = getattr(self, "_" + name)[0:N]
        return data

    def _restore_more(self, data):
        N = self.total_num_search
        for name in _TIMES:
            getattr(self, "_" + name)[0:N] = data[name]


class _DiscreteMultiHistory(_ActionColumn, _MultiObjectiveHistory):
    """Several objectives, discrete search space."""

    def __init__(self, num_objectives):
        self.num_objectives = num_objectives
        self._allocate()


class _RangeMultiHistory(_PointColumn, _MultiObjectiveHistory):
    """Several objectives, continuous search space."""

    def __init__(self, num_objectives, dim):
        self.dim = dim
        self.num_objectives = num_objectives
        self._allocate()
