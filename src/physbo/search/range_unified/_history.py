# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pickle
import copy

from .. import pareto

MAX_SEARCH = int(30000)


class History(object):
    def __init__(self, num_objectives, dim):
        self.dim = dim
        self.num_objectives = num_objectives
        self.pareto = pareto.Pareto(num_objectives=self.num_objectives)

        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros((MAX_SEARCH, self.num_objectives), dtype=float)
        self.action_X = np.zeros((MAX_SEARCH, self.dim), dtype=float)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

        self._time_total = np.zeros(MAX_SEARCH, dtype=float)
        self._time_update_predictor = np.zeros(MAX_SEARCH, dtype=float)
        self._time_get_action = np.zeros(MAX_SEARCH, dtype=float)
        self._time_run_simulator = np.zeros(MAX_SEARCH, dtype=float)

    @property
    def time_total(self):
        return copy.copy(self._time_total[0 : self.num_runs])

    @property
    def time_update_predictor(self):
        return copy.copy(self._time_update_predictor[0 : self.num_runs])

    @property
    def time_get_action(self):
        return copy.copy(self._time_get_action[0 : self.num_runs])

    @property
    def time_run_simulator(self):
        return copy.copy(self._time_run_simulator[0 : self.num_runs])

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
        Overwrite fx and action_X by t and action_X.

        Parameters
        ----------
        t: numpy.ndarray
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        action_X: numpy.ndarray
            N x d dimensional array. The input of each search candidate.
        time_total: numpy.ndarray
            N dimenstional array. The total elapsed time in each step.
            If None (default), filled by 0.0.
        time_update_predictor: numpy.ndarray
            N dimenstional array. The elapsed time for updating predictor (e.g., learning hyperparemters) in each step.
            If None (default), filled by 0.0.
        time_get_action: numpy.ndarray
            N dimenstional array. The elapsed time for getting next action in each step.
            If None (default), filled by 0.0.
        time_run_simulator: numpy.ndarray
            N dimenstional array. The elapsed time for running the simulator in each step.
            If None (default), filled by 0.0.
        Returns
        -------
        """
        t = np.array(t)

        if t.ndim == 1:
            N = 1
            if len(t) != self.num_objectives:
                raise ValueError("t does not match the number of objectives")
        else:
            N = t.shape[0]
            if t.shape[1] != self.num_objectives:
                raise ValueError("t does not match the number of objectives")

        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en, :] = t
        self.action_X[st:en, :] = action_X
        self.num_runs += 1
        self.total_num_search += N

        # update Pareto set
        self.pareto.update_front(t)

        if time_total is None:
            time_total = np.zeros(N, dtype=float)
        self._time_total[st:en] = time_total

        if time_update_predictor is None:
            time_update_predictor = np.zeros(N, dtype=float)
        self._time_update_predictor[st:en] = time_update_predictor

        if time_get_action is None:
            time_get_action = np.zeros(N, dtype=float)
        self._time_get_action[st:en] = time_get_action

        if time_run_simulator is None:
            time_run_simulator = np.zeros(N, dtype=float)
        self._time_run_simulator[st:en] = time_run_simulator

    def export_pareto_front(self):
        return self.pareto.export_front()

    def save(self, filename):
        N = self.total_num_search
        M = self.num_runs

        obj = {
            "num_runs": M,
            "total_num_search": N,
            "fx": self.fx[0:N, :],
            "action_X": self.action_X[0:N, :],
            "terminal_num_run": self.terminal_num_run[0:M],
            "pareto": self.pareto,
        }

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        M = int(data["num_runs"])
        N = int(data["total_num_search"])
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N, :] = data["fx"]
        self.action_X[0:N, :] = data["action_X"]
        self.terminal_num_run[0:M] = data["terminal_num_run"]
        self.pareto = data["pareto"]

    def show_search_results_mo(self, N, disp_pareto_set=False):
        n = self.total_num_search
        pset, step = self.pareto.export_front()

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
            print(
                "%04d-th step: f(x) = %s (X = %s)"
                % (n, str(self.fx[n - 1]), str(self.action_X[n - 1]))
            )

            msg_pareto_set_updated(indent=True)

        else:
            msg_pareto_set_updated()

            print("list of simulation results")
            st = self.total_num_search - N
            en = self.total_num_search
            for n in range(st, en):
                print("f(x) = %s (X = %s)" % (str(self.fx[n]), str(self.action_X[n])))
            print("\n")
