import numpy as np
import copy
import pickle

from .. import utility

MAX_SEARCH = int(30000)


class history:
    def __init__(self):
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros(MAX_SEARCH, dtype=float)
        self.chosen_actions = np.zeros(MAX_SEARCH, dtype=int)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

        self.time_total_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_update_predictor_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_get_action_ = np.zeros(MAX_SEARCH, dtype=float)
        self.time_run_simulator_ = np.zeros(MAX_SEARCH, dtype=float)

    @property
    def time_total(self):
        return copy.copy(self.time_total_[0:self.num_runs])

    @property
    def time_update_predictor(self):
        return copy.copy(self.time_update_predictor_[0:self.num_runs])

    @property
    def time_get_action(self):
        return copy.copy(self.time_get_action_[0:self.num_runs])

    @property
    def time_run_simulator(self):
        return copy.copy(self.time_run_simulator_[0:self.num_runs])

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
        Overwrite fx and chosen_actions by t and action.

        Parameters
        ----------
        t: numpy.ndarray
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        action: numpy.ndarray
            N dimensional array. The indexes of actions of each search candidate.
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
        self.chosen_actions[st:en] = action
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
        Export fx and actions at each sequence.
        (The total number of data is num_runs.)

        Returns
        -------
        best_fx: numpy.ndarray
        best_actions: numpy.ndarray
        """
        best_fx = np.zeros(self.num_runs, dtype=float)
        best_actions = np.zeros(self.num_runs, dtype=int)
        for n in range(self.num_runs):
            index = np.argmax(self.fx[0 : self.terminal_num_run[n]])
            best_actions[n] = self.chosen_actions[index]
            best_fx[n] = self.fx[index]

        return best_fx, best_actions

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
        best_actions = np.zeros(self.total_num_search, dtype=int)
        best_fx[0] = self.fx[0]
        best_actions[0] = self.chosen_actions[0]

        for n in range(1, self.total_num_search):
            if best_fx[n - 1] < self.fx[n]:
                best_fx[n] = self.fx[n]
                best_actions[n] = self.chosen_actions[n]
            else:
                best_fx[n] = best_fx[n - 1]
                best_actions[n] = best_actions[n - 1]

        return best_fx, best_actions

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
            chosen_actions=self.chosen_actions[0:N],
            terminal_num_run=self.terminal_num_run[0:M],
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
        M = data["num_runs"]
        N = data["total_num_search"]
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data["fx"]
        self.chosen_actions[0:N] = data["chosen_actions"]
        self.terminal_num_run[0:M] = data["terminal_num_run"]
