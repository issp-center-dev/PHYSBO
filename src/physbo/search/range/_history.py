# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import copy

from .. import utility

MAX_SEARCH = int(30000)


class History:
    def __init__(self, dim: int):
        self.dim = dim
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros(MAX_SEARCH, dtype=float)
        self.action_X = np.zeros((MAX_SEARCH, self.dim), dtype=float)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

        self.best_index = np.zeros(MAX_SEARCH, dtype=int)
        self.best_index[0] = 0

        self.time_total_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_update_predictor_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_get_action_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_run_simulator_ = np.zeros(MAX_SEARCH, dtype=float)

    @property
    def time_total(self):
        return copy.copy(self.time_total_[0 : self.num_runs])

    @property
    def time_update_predictor(self):
        return copy.copy(self.time_update_predictor_[0 : self.num_runs])

    @property
    def time_get_action(self):
        return copy.copy(self.time_get_action_[0 : self.num_runs])

    @property
    def time_run_simulator(self):
        return copy.copy(self.time_run_simulator_[0 : self.num_runs])

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
        N = utility.length_vector(t)
        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en] = t
        self.action_X[st:en, :] = action_X

        if st == 0:
            self.best_index[0] = 0
        else:
            for n in range(st, en):
                if self.fx[n] > self.fx[self.best_index[n - 1]]:
                    self.best_index[n] = n
                else:
                    self.best_index[n] = self.best_index[n - 1]

        self.num_runs += 1
        self.total_num_search += N

        if time_total is None:
            time_total = np.zeros(N, dtype=float)
        self.time_total_[st:en] = time_total

        if time_update_predictor is None:
            time_update_predictor = np.zeros(N, dtype=float)
        self.time_update_predictor_[st:en] = time_update_predictor

        if time_get_action is None:
            time_get_action = np.zeros(N, dtype=float)
        self.time_get_action_[st:en] = time_get_action

        if time_run_simulator is None:
            time_run_simulator = np.zeros(N, dtype=float)
        self.time_run_simulator_[st:en] = time_run_simulator

    def export_sequence_best_fx(self):
        """
        Export best fx and X at each sequence (each call of write function).

        Returns
        -------
        best_fx: numpy.ndarray (num_runs)
            The best fx at each sequence.
        best_X: numpy.ndarray (num_runs, dim)
            The best X at each sequence.
        """

        best_fx = np.zeros(self.num_runs, dtype=float)
        best_X = np.zeros((self.num_runs, self.dim), dtype=float)

        for r in range(self.num_runs):
            n = self.terminal_num_run[r] - 1
            best_fx[r] = self.fx[self.best_index[n]]
            best_X[r, :] = self.action_X[self.best_index[n], :]

        return best_fx, best_X

    def export_all_sequence_best_fx(self):
        """
        Export all fx and actions at each sequence.
        (The total number of data is total_num_research.)

        Returns
        -------
        best_fx: numpy.ndarray
        best_actions: numpy.ndarray
        """

        best_fx = np.zeros(self.total_num_search, dtype=float)
        best_X = np.zeros((self.total_num_search, self.dim), dtype=float)
        for n in range(self.total_num_search):
            best_fx[n] = self.fx[self.best_index[n]]
            best_X[n, :] = self.action_X[self.best_index[n], :]
        return best_fx, best_X

    def save(self, filename):
        """
        Save the information of the history.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
        Returns
        -------

        """
        N = self.total_num_search
        M = self.num_runs
        np.savez_compressed(
            filename,
            num_runs=M,
            total_num_search=N,
            fx=self.fx[0:N],
            action_X=self.action_X[0:N, :],
            best_index=self.best_index[0:N],
            terminal_num_run=self.terminal_num_run[0:M],
            time_total=self.time_total_[0:N],
            time_update_predictor=self.time_update_predictor_[0:N],
            time_get_action=self.time_get_action_[0:N],
            time_run_simulator=self.time_run_simulator_[0:N],
        )

    def load(self, filename):
        """
        Load the information of the history.

        Parameters
        ----------
        filename: str
            The name of the file which stores the information of the history
        Returns
        -------

        """
        data = np.load(filename)
        M = int(data["num_runs"])
        N = int(data["total_num_search"])
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data["fx"]
        self.action_X[0:N, :] = data["action_X"]
        self.terminal_num_run[0:M] = data["terminal_num_run"]
        self.best_index[0:N] = data["best_index"]
        self.time_total_[0:N] = data["time_total"]
        self.time_update_predictor_[0:N] = data["time_update_predictor"]
        self.time_get_action_[0:N] = data["time_get_action"]
        self.time_run_simulator_[0:N] = data["time_run_simulator"]

    def show_search_results(self, N):
        n = self.total_num_search
        index = np.argmax(self.fx[0:n])

        if N == 1:
            print(
                f"{n:04d}-th step: f(x) = {self.fx[n - 1]:.6f} (action={self.action_X[n - 1, :]})"
            )
            print(
                f"   current best f(x) = {self.fx[index]:.6f} (best action={self.action_X[index, :]}) \n"
            )
        else:
            print(
                f"current best f(x) = {self.fx[index]:.6f} (best action={self.action_X[index, :]})"
            )

            print("list of simulation results")
            st = self.total_num_search - N
            en = self.total_num_search
            for n in range(st, en):
                print(f"f(x)={self.fx[n]:.6f} (action = {self.action_X[n, :]})")
            print("\n")
