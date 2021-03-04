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


def run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)


class policy(discrete.policy):
    def __init__(
        self, test_X, num_objectives, comm=None, config=None, initial_actions=None
    ):
        self.num_objectives = num_objectives
        self.history = history(num_objectives=self.num_objectives)

        self.training_list = [variable() for i in range(self.num_objectives)]
        self.predictor_list = [None for i in range(self.num_objectives)]
        self.test_list = [self._set_test(test_X) for i in range(self.num_objectives)]
        self.new_data_list = [None for i in range(self.num_objectives)]

        self.actions = np.arange(0, test_X.shape[0])
        self.config = self._set_config(config)

        self.TS_candidate_num = None

        if initial_actions is not None:
            self.actions = sorted(list(set(self.actions) - set(initial_actions)))

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

            self.new_data_list[i] = variable(X, t[:, i], Z)
            self.training_list[i].add(X=X, t=t[:, i], Z=Z)

    def _switch_model(self, i):
        self.training = self.training_list[i]
        self.predictor = self.predictor_list[i]
        self.test = self.test_list[i]
        self.new_data = self.new_data_list[i]

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

        if int(max_num_probes) * N > len(self.actions):
            raise ValueError(
                "max_num_probes * num_search_each_probe must \
                be smaller than the length of candidates"
            )

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            if is_disp and N > 1:
                utility.show_start_message_multi_search_mo(
                    self.history.num_runs, "random"
                )

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t = run_simulator(simulator, action, self.mpicomm)
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
            self.predictor = predictor_list

        N = int(num_search_each_probe)

        for n in range(max_num_probes):

            if utility.is_learning(n, interval):
                for i in range(self.num_objectives):
                    self._switch_model(i)
                    self.predictor.fit(self.training, num_rand_basis)
                    self.test.Z = self.predictor.get_basis(self.test.X)
                    self.training.Z = self.predictor.get_basis(self.training.X)
                    self.predictor.prepare(self.training)
            else:
                for i in range(self.num_objectives):
                    self._switch_model(i)
                    self.predictor.update(self.training, self.new_data)

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search_mo(self.history.num_runs, score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self.get_actions(score, N, K, alpha)

            if len(action) == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                return copy.deepcopy(self.history)

            if simulator is None:
                return action

            t = run_simulator(simulator, action, self.mpicomm)
            self.write(action, t)

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )

        return copy.deepcopy(self.history)

    def get_actions(self, mode, N, K, alpha):
        f = self.get_score(
            mode=mode,
            predictor_list=self.predictor_list,
            training_list=self.training_list,
            pareto=self.history.pareto,
            alpha=alpha,
        )
        champion, local_champion, local_index = self._find_champion(f)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self.delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f = self.get_score(
                mode=mode,
                predictor_list=self.predictor_list,
                training_list=self.training_list,
                pareto=self.history.pareto,
                alpha=alpha,
            )
            champion, local_champion, local_index = self._find_champion(f)
            if champion == -1:
                break
            chosen_actions.append(champion)
            if champion == local_champion:
                self.actions = self.delete_actions(local_index)
        return np.array(chosen_actions)

    def get_score(self, mode, predictor_list, training_list, pareto, alpha=1):
        actions = self.actions
        test = self.test.get_subset(actions)

        if test.X.shape[0] == 0:
            return np.zeros(0)

        if mode == "EHVI":
            fmean, fstd = self.get_fmean_fstd(predictor_list, training_list, test)
            f = search_score.EHVI(fmean, fstd, pareto)
        elif mode == "HVPI":
            fmean, fstd = self.get_fmean_fstd(predictor_list, training_list, test)
            f = search_score.HVPI(fmean, fstd, pareto)
        elif mode == "TS":
            f = search_score.TS(
                predictor_list,
                training_list,
                test,
                alpha,
                reduced_candidate_num=self.TS_candidate_num,
            )
        else:
            raise NotImplementedError("mode must be EHVI, HVPI or TS.")
        return f

    def get_fmean_fstd(self, predictor_list, training_list, test):
        fmean = [
            predictor.get_post_fmean(training, test)
            for predictor, training in zip(predictor_list, training_list)
        ]
        fcov = [
            predictor.get_post_fcov(training, test)
            for predictor, training in zip(predictor_list, training_list)
        ]

        # shape: (N, n_obj)
        fmean = np.array(fmean).T
        fstd = np.sqrt(np.array(fcov)).T
        return fmean, fstd

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
