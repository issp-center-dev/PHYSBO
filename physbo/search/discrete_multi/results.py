import numpy as np
import pickle

from .. import pareto

MAX_SEARCH = int(30000)


class history(object):
    def __init__(self, num_objectives):
        self.num_objectives = num_objectives
        self.pareto = pareto.Pareto(num_objectives=self.num_objectives)

        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros((MAX_SEARCH, self.num_objectives), dtype=float)
        self.chosen_actions = np.zeros(MAX_SEARCH, dtype=int)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

    def write(self, t, action):
        t = np.array(t)
        action = np.array(action)

        if t.ndim == 1:
            N = 1
            if len(t) != self.num_objectives:
                raise ValueError('t does not match the number of objectives')
        else:
            N = t.shape[0]
            if t.shape[1] != self.num_objectives:
                raise ValueError('t does not match the number of objectives')

        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en] = t
        self.chosen_actions[st:en] = action
        self.num_runs += 1
        self.total_num_search += N

        # update Pareto set
        self.pareto.update_front(t)

    def export_pareto_front(self):
        return self.pareto.export_front()

    def save(self, filename):
        N = self.total_num_search
        M = self.num_runs

        obj = {"num_runs": M, "total_num_search": N,
               "fx": self.fx[0:N], "chosen_actions": self.chosen_actions[0:N],
               "terminal_num_run": self.terminal_num_run[0:M],
               "pareto": self.pareto}

        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        M = data['num_runs']
        N = data['total_num_search']
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data['fx']
        self.chosen_actions[0:N] = data['chosen_actions']
        self.terminal_num_run[0:M] = data['terminal_num_run']
        self.pareto = data['pareto']
