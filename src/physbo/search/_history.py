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

The arrays of a history (``fx``, the points, ``terminal_num_run``, ...) have
as many entries as there are observations or steps. They are views of
storage that grows as observations are written, so an array obtained before
a call of ``write`` does not show what is written afterwards.
"""

import os
import pickle

import numpy as np

from . import pareto
from . import utility

#: the number of observations and of steps a new history has room for; the
#: storage is doubled whenever it is full
_INITIAL_ROOM = 64

#: the elapsed times recorded with each observation, and what they measure
_TIMES = {
    "time_total": "the whole step",
    "time_update_predictor": "updating the predictor",
    "time_get_action": "choosing the next points",
    "time_run_simulator": "running the simulator",
}


def _elapsed_time(name):
    """The property that returns the times of the steps.

    The times are stored with each observation, those of a step with the
    same value; the time of a step is read from its first observation (and
    is zero for a step without observations).
    """

    def getter(self):
        end = self.terminal_num_run[0 : self.num_runs]
        start = np.zeros_like(end)
        start[1:] = end[:-1]
        times = np.zeros(self.num_runs, dtype=float)
        filled = end > start
        times[filled] = getattr(self, "_" + name)[start[filled]]
        return times

    return property(getter, doc=f"The elapsed time of {_TIMES[name]}, for each step.")


def _recorded(storage, count, doc):
    """The property that gives the used part of an array of the storage: its
    first ``count`` entries (the name of the counter), as a view.

    Assigning to the property assigns these entries; it does not replace the
    storage.
    """

    def getter(self):
        return getattr(self, storage)[0 : getattr(self, count)]

    def setter(self, values):
        getter(self)[...] = values

    return property(getter, setter, doc=doc)


def _grown(array, rows):
    """The array with room for ``rows`` rows: itself, or a copy at least
    twice as long."""
    if rows <= len(array):
        return array
    grown = np.zeros((max(rows, 2 * len(array)),) + array.shape[1:], array.dtype)
    grown[0 : len(array)] = array
    return grown


class _HistoryBase:
    """What the histories of all search modules have in common.

    A class of a search module combines a class of this kind with a
    description of the column of the points (:class:`_ActionColumn` or
    :class:`_PointColumn`), and calls :meth:`_allocate` in its constructor.

    The storage has one array per quantity, with room for more entries than
    are used: ``_fx``, ``_recorded`` (the points) and the elapsed times have
    one row per observation, ``_terminal`` one entry per step. The public
    arrays are the used parts of them.
    """

    #: the arrays of the storage with one row per observation
    _per_observation = ("_fx", "_recorded") + tuple("_" + name for name in _TIMES)

    def _allocate(self):
        """Make the storage of an empty history."""
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self._set_storage(
            np.zeros((0,) + self._fx_shape(), dtype=float),
            np.zeros((0,) + self._column_shape(), dtype=self._column_dtype),
            np.zeros(0, dtype=int),
        )
        self._reserve(_INITIAL_ROOM, _INITIAL_ROOM)

    def _set_storage(self, fx, points, terminal_num_run):
        """Take the observations given as arrays (of exactly their number)
        as the storage, with elapsed times of zero."""
        self._fx = fx
        self._recorded = points
        self._terminal = terminal_num_run
        for name in _TIMES:
            setattr(self, "_" + name, np.zeros(len(fx), dtype=float))

    def _reserve(self, num_observations, num_steps):
        """Make room for this many observations and steps."""
        for name in self._per_observation:
            setattr(self, name, _grown(getattr(self, name), num_observations))
        self._terminal = _grown(self._terminal, num_steps)

    def _fx_shape(self):
        """The shape of the objective value(s) of one observation."""
        raise NotImplementedError

    fx = _recorded(
        "_fx",
        "total_num_search",
        "The objective values of the observations (total_num_search rows).",
    )
    terminal_num_run = _recorded(
        "_terminal",
        "num_runs",
        "The number of observations at the end of each step (num_runs entries).",
    )
    #: the points of the observations (chosen_actions or action_X)
    _points = _recorded("_recorded", "total_num_search", None)

    def __getstate__(self):
        # the used part of the storage, so that a pickle or a copy does not
        # depend on the room the history happens to have
        state = dict(self.__dict__)
        for name in self._per_observation:
            state[name] = state[name][0 : self.total_num_search].copy()
        state["_terminal"] = self._terminal[0 : self.num_runs].copy()
        return state

    def __setstate__(self, state):
        if "fx" in state:
            state = self._from_arrays_of_fixed_length(state)
        self.__dict__.update(state)

    def _from_arrays_of_fixed_length(self, state):
        """The state of a history pickled by a version in which the arrays
        had 30,000 rows and were attributes under their public names (the
        elapsed times as ``_time_total`` or, before that in the
        single-objective classes, ``time_total_``, and so on)."""
        state = dict(state)
        N = state["total_num_search"]
        M = state["num_runs"]
        new = {
            "_fx": state.pop("fx")[0:N].copy(),
            "_recorded": state.pop(self._column_name)[0:N].copy(),
            "_terminal": state.pop("terminal_num_run")[0:M].copy(),
        }
        for name in _TIMES:
            earlier = name + "_" if name + "_" in state else "_" + name
            new["_" + name] = state.pop(earlier)[0:N].copy()
        # it is made again from fx (single objective)
        state.pop("best_index", None)
        state.update(new)
        return state

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
        return utility.finite_mask(self.fx)

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
        mask = self.valid_mask
        return self._points[mask], self.fx[mask]

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
        # everything is checked before anything is written
        t, points, N = self._check_observations(t, points)
        times = (time_total, time_update_predictor, time_get_action, time_run_simulator)
        times = [self._check_times(name, v, N) for name, v in zip(_TIMES, times)]
        st = self.total_num_search
        en = st + N

        self._reserve(en, self.num_runs + 1)
        self._fx[st:en] = t
        self._recorded[st:en] = points
        for name, values in zip(_TIMES, times):
            getattr(self, "_" + name)[st:en] = values
        self._terminal[self.num_runs] = en
        self.num_runs += 1
        self.total_num_search = en

        self._update_summary(st, en, t)

    def _check_observations(self, t, points):
        """Check the arguments of ``write`` and give them the shape of the
        arrays: one row per observation.

        Returns
        -------
        The objective values, the points, and the number of observations.

        Raises
        ------
        ValueError
            If an argument does not have a shape the class accepts, or if
            the numbers of objective values and of points differ.
        """
        t = self._as_objective_values(np.asarray(t, dtype=float))
        points = self._as_points(np.asarray(points))
        if t.shape[0] != points.shape[0]:
            raise ValueError(
                "The numbers of objective values and of points must be the same:"
                f" {t.shape[0]} != {points.shape[0]}"
            )
        return t, points, t.shape[0]

    def _as_objective_values(self, t):
        """The objective values with one row per observation; ValueError if
        the array does not have a shape the class accepts."""
        raise NotImplementedError

    def _as_points(self, points):
        """The points with one row per observation; ValueError if the array
        does not have a shape the class accepts."""
        raise NotImplementedError

    @staticmethod
    def _check_times(name, values, N):
        """An elapsed time for each of N observations: zero if not given, and
        one value for all of them or one each otherwise."""
        if values is None:
            return np.zeros(N, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape not in ((), (1,), (N,)):
            raise ValueError(
                f"{name} must have one value, or one for each of the {N}"
                f" observations: its shape is {values.shape}"
            )
        return np.broadcast_to(values, (N,))

    def _update_summary(self, st, en, t):
        """Bring the summary up to date with the observations st to en."""
        raise NotImplementedError

    def _saved_arrays(self):
        """The entries of the file of ``save``."""
        N = self.total_num_search
        data = {
            "num_runs": self.num_runs,
            "total_num_search": N,
            "fx": self.fx,
            self._column_name: self._points,
        }
        data.update(self._saved_summary())
        data["terminal_num_run"] = self.terminal_num_run
        # with one entry per observation, as they are stored
        for name in _TIMES:
            data[name] = getattr(self, "_" + name)[0:N]
        return data

    def _saved_summary(self):
        """What the class saves of its summary."""
        raise NotImplementedError

    def load(self, filename):
        """
        Load the information of the history, in place of what it holds.

        The file is one written by ``save``, of this or of an earlier
        version. If there is no file of this name but one with ".npz"
        appended (as numpy does when saving), that one is read.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history

        Raises
        ------
        ValueError
            If the file is not that of a history of this kind (an entry is
            missing, or the arrays do not have the shape of this history).
            The history is left as it was.
        """
        data = self._read_saved(filename)
        try:
            M = int(data["num_runs"])
            N = int(data["total_num_search"])
            arrays = {
                "fx": ((N,) + self._fx_shape(), float),
                self._column_name: ((N,) + self._column_shape(), self._column_dtype),
                "terminal_num_run": ((M,), int),
            }
            # a file of an earlier version may not have the times
            arrays.update({name: ((N,), float) for name in _TIMES if name in data})
            loaded = {}
            for name, (shape, dtype) in arrays.items():
                loaded[name] = np.array(data[name], dtype=dtype)
                if loaded[name].shape != shape:
                    raise ValueError(
                        f"{name} has the shape {loaded[name].shape}, not {shape}"
                    )
        except (KeyError, ValueError) as error:
            what = f"no entry {error}" if isinstance(error, KeyError) else str(error)
            raise ValueError(
                f"{filename} is not the file of a history of this kind"
                f" ({type(self).__module__}.{type(self).__name__}): {what}"
            ) from None

        self.num_runs = M
        self.total_num_search = N
        self._set_storage(
            loaded["fx"], loaded[self._column_name], loaded["terminal_num_run"]
        )
        for name in _TIMES:
            if name in loaded:
                setattr(self, "_" + name, loaded[name])
        self._restore_summary(data)

    #: whether ``load`` also reads the pickles that the multi-objective
    #: classes used to save
    _loads_pickles = False

    def _read_saved(self, filename):
        """The entries of a file written by ``save``, as a dict."""
        if not isinstance(filename, (str, os.PathLike)):
            # an open file: numpy reads it
            with np.load(filename) as archive:
                return dict(archive)

        path = os.fspath(filename)
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path += ".npz"
        if self._loads_pickles:
            # an archive of numpy is a zip file
            with open(path, "rb") as f:
                is_archive = f.read(2) == b"PK"
            if not is_archive:
                with open(path, "rb") as f:
                    return pickle.load(f)
        with np.load(path) as archive:
            return dict(archive)

    def _restore_summary(self, data):
        """Make the summary of the observations that were loaded, with what
        the file holds of it."""
        raise NotImplementedError


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

    chosen_actions = _recorded(
        "_recorded",
        "total_num_search",
        "The indexes of the actions of the observations (total_num_search entries).",
    )

    def _as_points(self, points):
        if points.ndim == 0:
            points = points.reshape(1)
        if points.ndim != 1:
            raise ValueError(
                "action must be the index of an action or N of them:"
                f" its shape is {points.shape}"
            )
        return points

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
            (to be maximized). For a single objective, an N dimensional or
            an N x 1 array; for several objectives, an N x num_objectives
            array, or an array of num_objectives values for one observation.
        action: numpy.ndarray
            N dimensional array. The indexes of the actions of the
            observations (or one index, for one observation).
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

        Each elapsed time is that of the step, given once or repeated for
        each observation.

        Raises
        ------
        ValueError
            If the numbers of objective values and of points differ, or if
            an argument does not have one of the shapes above. Nothing is
            written in this case.
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

    action_X = _recorded(
        "_recorded",
        "total_num_search",
        "The inputs of the observations (total_num_search x dim).",
    )

    def _as_points(self, points):
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(
                f"action_X must be a point of {self.dim} coordinates or an"
                f" N x {self.dim} array: its shape is {points.shape}"
            )
        return points

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
            (to be maximized). For a single objective, an N dimensional or
            an N x 1 array; for several objectives, an N x num_objectives
            array, or an array of num_objectives values for one observation.
        action_X: numpy.ndarray
            N x d dimensional array. The inputs of the observations (or an
            array of d values, for one observation).
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

        Each elapsed time is that of the step, given once or repeated for
        each observation.

        Raises
        ------
        ValueError
            If the numbers of objective values and of points differ, or if
            an argument does not have one of the shapes above. Nothing is
            written in this case.
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

    _per_observation = _HistoryBase._per_observation + ("_best_index",)

    def _set_storage(self, fx, points, terminal_num_run):
        super()._set_storage(fx, points, terminal_num_run)
        self._best_index = np.full(len(fx), -1, dtype=int)
        self._update_best_index(0, len(fx))

    def _fx_shape(self):
        return ()

    best_index = _recorded(
        "_best_index",
        "total_num_search",
        "The index of the best valid observation after each observation, -1"
        " until the first valid one (total_num_search entries).",
    )

    def _as_objective_values(self, t):
        shape = t.shape
        if t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(
                "t must be N values of the objective, or an N x 1 array:"
                f" its shape is {shape}"
            )
        return t

    # the elapsed times with one entry per observation, under the names
    # these arrays had as attributes (kept for compatibility)
    time_total_ = _recorded("_time_total", "total_num_search", None)
    time_update_predictor_ = _recorded(
        "_time_update_predictor", "total_num_search", None
    )
    time_get_action_ = _recorded("_time_get_action", "total_num_search", None)
    time_run_simulator_ = _recorded("_time_run_simulator", "total_num_search", None)

    def __setstate__(self, state):
        super().__setstate__(state)
        if "_best_index" not in state:
            # pickled by a version with arrays of fixed length
            N = self.total_num_search
            self._best_index = np.full(N, -1, dtype=int)
            self._update_best_index(0, N)

    def _update_summary(self, st, en, t):
        self._update_best_index(st, en)

    def _update_best_index(self, st, en):
        """
        Update best_index[st:en] from fx.

        Failed (non-finite) observations are skipped, and best_index stays -1
        until the first valid one. Of equal values the earlier one is kept.
        """
        fx = self._fx[st:en]
        valid = np.isfinite(fx)
        previous = self._best_index[st - 1] if st > 0 else -1
        best_so_far = self._fx[previous] if previous >= 0 else -np.inf

        # the best value before each observation; an observation is the new
        # best if it is valid and larger than that
        values = np.where(valid, fx, -np.inf)
        before = np.maximum.accumulate(np.concatenate(([best_so_far], values)))[:-1]
        index = np.where(valid & (values > before), np.arange(st, en), previous)
        # the indexes of the new bests increase, so the last one so far is
        # the largest
        self._best_index[st:en] = np.maximum.accumulate(index)

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
        last = self.terminal_num_run - 1
        # a step before the first observation has nothing to refer to
        index = np.full(len(last), -1, dtype=int)
        index[last >= 0] = self.best_index[last[last >= 0]]
        return self._best(index)

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
        return self._best(self.best_index)

    def save(self, filename):
        """
        Save the information of the history, as an archive of numpy
        (``numpy.savez_compressed``).

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
            (numpy appends ".npz" if the name does not end with it)
        """
        np.savez_compressed(filename, **self._saved_arrays())

    def _saved_summary(self):
        return {"best_index": self.best_index}

    def _restore_summary(self, data):
        # best_index has been made from fx with the storage: the saved one
        # may refer to a failed observation if the file was written by an
        # older version
        pass

    def show_search_results(self, N):
        """Print the last N observations and the best value so far."""
        n = self.total_num_search
        N = min(N, n)
        fmt = self._format_point
        index = self.best_index[n - 1] if n > 0 else -1
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

    def _as_objective_values(self, t):
        shape = t.shape
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if t.ndim != 2 or t.shape[1] != self.num_objectives:
            raise ValueError(
                f"t does not match the number of objectives"
                f" ({self.num_objectives}): its shape is {shape}"
            )
        return t

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

    _loads_pickles = True

    def save(self, filename):
        """
        Save the information of the history, as an archive of numpy
        (``numpy.savez_compressed``). The reference points of the Pareto
        front are saved with it; the front itself is made again from the
        observations when the file is loaded.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the
            history. It is used as it is (".npz" is not appended).
        """
        with open(filename, "wb") as f:
            np.savez_compressed(f, **self._saved_arrays())

    def _saved_summary(self):
        # the reference points that are set, and whether they were estimated
        # from the front (then they follow it) or given
        data = {}
        for side in ("min", "max"):
            reference = getattr(self.pareto, "reference_" + side)
            if reference is None:
                continue
            estimated = getattr(self.pareto, "_estimated_reference_" + side, None)
            data["reference_" + side] = np.asarray(reference, dtype=float)
            data["reference_" + side + "_estimated"] = reference is estimated
        return data

    def _restore_summary(self, data):
        if "pareto" in data:
            # a pickle of an earlier version, with the Pareto object
            self.pareto = data["pareto"]
        else:
            self.pareto = pareto.Pareto(num_objectives=self.num_objectives)
            for side in ("min", "max"):
                if "reference_" + side not in data:
                    continue
                reference = np.array(data["reference_" + side], dtype=float)
                estimated = bool(data["reference_" + side + "_estimated"])
                setattr(self.pareto, "reference_" + side, reference)
                setattr(
                    self.pareto,
                    "_estimated_reference_" + side,
                    reference if estimated else None,
                )
        # the front is made from fx: a saved one may contain failed
        # observations if the file was written by an older version
        self.pareto.rebuild_front(self.fx, self.terminal_num_run)

    def show_search_results_mo(self, N, disp_pareto_set=False):
        """Print the last N observations, and the Pareto front if they have
        changed it (the front itself with ``disp_pareto_set``, else its
        size)."""
        n = self.total_num_search
        N = min(N, n)
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


class _RangeHistory(_PointColumn, _SingleObjectiveHistory):
    """Single objective, continuous search space."""

    def __init__(self, dim: int):
        self.dim = dim
        self._allocate()


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
