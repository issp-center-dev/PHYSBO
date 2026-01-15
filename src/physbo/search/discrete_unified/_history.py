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
    def __init__(self, num_objectives):
        self.num_objectives = num_objectives
        self.pareto = pareto.Pareto(num_objectives=self.num_objectives)

        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros((MAX_SEARCH, self.num_objectives), dtype=float)
        self.chosen_actions = np.zeros(MAX_SEARCH, dtype=int)
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
        action,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        t = np.array(t)
        action = np.array(action)

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
        self.fx[st:en] = t
        self.chosen_actions[st:en] = action
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
            "fx": self.fx[0:N],
            "chosen_actions": self.chosen_actions[0:N],
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
        self.fx[0:N] = data["fx"]
        self.chosen_actions[0:N] = data["chosen_actions"]
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
                "%04d-th step: f(x) = %s (action = %d)"
                % (n, str(self.fx[n - 1]), self.chosen_actions[n - 1])
            )

            msg_pareto_set_updated(indent=True)

        else:
            msg_pareto_set_updated()

            print("list of simulation results")
            st = self.total_num_search - N
            en = self.total_num_search
            for n in range(st, en):
                print(
                    "f(x) = %s (action = %d)"
                    % (str(self.fx[n]), self.chosen_actions[n])
                )
            print("\n")
