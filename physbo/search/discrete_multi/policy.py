import numpy as np
import copy
import pickle as pickle

from .results import history
from .. import discrete
from .. import utility
from .. import score_multi as search_score
from ...gp import predictor as gp_predictor
from ...blm import predictor as blm_predictor
from ...misc import set_config
from ...variable import variable

from typing import List, Optional


class policy(discrete.policy):
    new_data_list: List[Optional[variable]]

    def __init__(
        self, test_X, num_objectives, comm=None, config=None, initial_data=None
    ):
        self.num_objectives = num_objectives
        self.history = history(num_objectives=self.num_objectives)

        self.training_list = [variable() for _ in range(self.num_objectives)]
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.test_list = [
            self._make_variable_X(test_X) for _ in range(self.num_objectives)
        ]
        self.new_data_list = [None for _ in range(self.num_objectives)]

        self.actions = np.arange(0, test_X.shape[0])
        if config is None:
            self.config = set_config()
        else:
            self.config = config

        self.TS_candidate_num = None

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
                raise RuntimeError(msg)
            actions, fs = initial_data
            if fs.shape[1] != self.num_objectives:
                msg = "ERROR: initial_data[1].shape[1] != num_objectives"
                raise RuntimeError(msg)
            if len(actions) != fs.shape[0]:
                msg = "ERROR: len(initial_data[0]) != initial_data[1].shape[0]"
                raise RuntimeError(msg)
            self.write(actions, fs)
            self.actions = sorted(list(set(self.actions) - set(actions)))

        if comm is None:
            self.mpicomm = None
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpicomm = comm
            self.mpisize = comm.size
            self.mpirank = comm.rank
            self.actions = np.array_split(self.actions, self.mpisize)[self.mpirank]

    def write(self, action, t, X=None):
        self.history.write(t, action)
        action = np.array(action)
        t = np.array(t)

        for i in range(self.num_objectives):
            test = self.test_list[i]
            predictor = self.predictor_list[i]

            if X is None:
                X = test.X[action, :]
                Z = test.Z[action, :] if test.Z is not None else None
            else:
                Z = predictor.get_basis(X) if predictor is not None else None

            if self.new_data_list[i] is None:
                self.new_data_list[i] = variable(X, t[:, i], Z)
            else:
                self.new_data_list[i].add(X=X, t=t[:, i], Z=Z)
            self.training_list[i].add(X=X, t=t[:, i], Z=Z)

    def _model(self, i):
        training = self.training_list[i]
        predictor = self.predictor_list[i]
        test = self.test_list[i]
        new_data = self.new_data_list[i]
        return {"training": training, "predictor": predictor, "test": test, "new_data": new_data}

    def random_search(
        self,
        max_num_probes,
        num_search_each_probe=1,
        simulator=None,
        is_disp=True,
        disp_pareto_set=False,
    ):

        if self.mpirank != 0:
            is_disp = False

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            if is_disp and N > 1:
                utility.show_start_message_multi_search_mo(
                    self.history.num_runs, "random"
                )

            action = self._get_random_action(N)

            if simulator is None:
                return action

            t = _run_simulator(simulator, action, self.mpicomm)
            self.write(action, t)

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )

        return copy.deepcopy(self.history)

    def bayes_search(
        self,
        training_list=None,
        max_num_probes=None,
        num_search_each_probe=1,
        predictor_list=None,
        is_disp=True,
        disp_pareto_set=False,
        simulator=None,
        score="HVPI",
        interval=0,
        num_rand_basis=0,
    ):

        if self.mpirank != 0:
            is_disp = False

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        if training_list is not None:
            self.training_list = training_list

        if predictor_list is None:
            if is_rand_expans:
                self.predictor_list = [
                    blm_predictor(self.config) for i in range(self.num_objectives)
                ]
            else:
                self.predictor_list = [
                    gp_predictor(self.config) for i in range(self.num_objectives)
                ]
        else:
            self.predictor_list = predictor_list

        if max_num_probes == 0 and interval >= 0:
            self._learn_hyperparameter(num_rand_basis)

        N = int(num_search_each_probe)

        for n in range(max_num_probes):

            if utility.is_learning(n, interval):
                self._learn_hyperparameter(num_rand_basis)
            else:
                self._update_predictor()

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search_mo(self.history.num_runs, score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self._get_actions(score, N, K, alpha)

            if len(action) == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                return copy.deepcopy(self.history)

            if simulator is None:
                return action

            t = _run_simulator(simulator, action, self.mpicomm)
            self.write(action, t)

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )
        self._update_predictor()
        return copy.deepcopy(self.history)

    def _get_actions(self, mode, N, K, alpha):
        f = self.get_score(mode=mode, alpha=alpha, parallel=False)
        champion, local_champion, local_index = self._find_champion(f)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f = self.get_score(mode=mode, alpha=alpha, parallel=False)
            champion, local_champion, local_index = self._find_champion(f)
            if champion == -1:
                break
            chosen_actions.append(champion)
            if champion == local_champion:
                self.actions = self._delete_actions(local_index)
        return np.array(chosen_actions)

    def get_post_fmean(self, xs):
        X = self._make_variable_X(xs)
        predictor_list = self.predictor_list[:]
        if predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_post_fmean()")
            for i in range(self.num_objectives):
                predictor_list[i] = gp_predictor(self.config)
                predictor_list[i].fit(self.training_list[i], 0)
                predictor_list[i].prepare(self.training_list[i])
        fmean = [
            predictor.get_post_fmean(training, X)
            for predictor, training in zip(predictor_list, self.training_list)
        ]
        return np.array(fmean).T

    def get_post_fcov(self, xs):
        X = self._make_variable_X(xs)
        predictor_list = self.predictor_list[:]
        if predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_post_fcov()")
            for i in range(self.num_objectives):
                predictor_list[i] = gp_predictor(self.config)
                predictor_list[i].fit(self.training_list[i], 0)
                predictor_list[i].prepare(self.training_list[i])
        fcov = [
            predictor.get_post_fcov(training, X)
            for predictor, training in zip(predictor_list, self.training_list)
        ]
        return np.array(fcov).T

    def get_score(
        self,
        mode,
        actions=None,
        xs=None,
        predictor_list=None,
        training_list=None,
        pareto=None,
        parallel=True,
        alpha=1,
    ):
        if predictor_list is None:
            predictor_list = self.predictor_list
        if training_list is None:
            training_list = self.training_list
        if pareto is None:
            pareto = self.history.pareto

        if training_list[0].X is None or training_list[0].X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_score()")
            for i in range(self.num_objectives):
                predictor_list[i] = gp_predictor(self.config)
                predictor_list[i].fit(training_list[i], 0)
                predictor_list[i].prepare(training_list[i])

        if xs is not None:
            if actions is not None:
                raise RuntimeError("ERROR: both actions and xs are given")
            if isinstance(xs, variable):
                test = xs
            else:
                test = variable(X=xs)
            if parallel and self.mpisize > 1:
                actions = np.array_split(np.arange(test.X.shape[0]), self.mpisize)
                test = test.get_subset(actions[self.mpirank])
        else:
            if actions is None:
                actions = self.actions
            else:
                if isinstance(actions, int):
                    actions = [actions]
                if parallel and self.mpisize > 1:
                    actions = np.array_split(actions, self.mpisize)[self.mpirank]
            test = self.test_list[0].get_subset(actions)

        f = search_score.score(
            mode,
            predictor_list=predictor_list,
            training_list=training_list,
            test=test,
            pareto=pareto,
            reduced_candidate_num=self.TS_candidate_num,
            alpha=alpha,
        )
        if parallel and self.mpisize>1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f

    def save(self, file_history, file_training_list=None, file_predictor_list=None):
        if self.mpirank == 0:
            self.history.save(file_history)
            if file_training_list is not None:
                self.save_training_list(file_training_list)
            if file_predictor_list is not None:
                self.save_predictor_list(file_predictor_list)

    def load(self, file_history, file_training_list=None, file_predictor_list=None):
        self.history.load(file_history)

        if file_training_list is None:
            N = self.history.total_num_search
            X = self.test_list[0].X[self.history.chosen_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training_list = [
                variable(X=X, t=t[:, i]) for i in range(self.num_objectives)
            ]
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

    def save_predictor_list(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.predictor_list, f, 2)

    def save_training_list(self, file_name):
        obj = [
            {"X": training.X, "t": training.t, "Z": training.Z}
            for training in self.training_list
        ]
        with open(file_name, "wb") as f:
            pickle.dump(obj, f, 2)

    def load_predictor_list(self, file_name):
        with open(file_name, "rb") as f:
            self.predictor_list = pickle.load(f)

    def load_training_list(self, file_name):
        with open(file_name, "rb") as f:
            data_list = pickle.load(f)

        self.training_list = [variable() for i in range(self.num_objectives)]
        for data, training in zip(data_list, self.training_list):
            training.X = data["X"]
            training.t = data["t"]
            training.Z = data["Z"]

    def _learn_hyperparameter(self, num_rand_basis):
        for i in range(self.num_objectives):
            m = self._model(i)
            predictor = m["predictor"]
            training = m["training"]
            test = m["test"]

            predictor.fit(training, num_rand_basis)
            test.Z = predictor.get_basis(test.X)
            training.Z = predictor.get_basis(training.X)
            predictor.prepare(training)
            self.new_data_list[i] = None
            # self.predictor_list[i].fit(self.training_list[i], num_rand_basis)
            # self.test_list[i].Z = self.predictor_list[i].get_basis(self.test_list[i].X)
            # self.training_list[i].Z = self.predictor_list[i].get_basis(self.training_list[i].X)
            # self.predictor_list[i].prepare(self.training_list[i])
            # self.new_data_list[i] = None

    def _update_predictor(self):
        for i in range(self.num_objectives):
            if self.new_data_list[i] is not None:
                self.predictor_list[i].update(self.training_list[i], self.new_data_list[i])
                self.new_data_list[i] = None


def _run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)
