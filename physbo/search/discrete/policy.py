import numpy as np
import copy
import physbo.misc
import pickle as pickle
from .results import history, history_mo
from .. import utility
from ...variable import variable
from ..call_simulator import call_simulator, call_simulator_mo
from ...gp import predictor as gp_predictor
from ...blm import predictor as blm_predictor
import physbo.search.score

MAX_SEACH = int(20000)


class policy:
    def __init__(self, test_X, config=None):
        """

        Parameters
        ----------
        test_X: numpy.ndarray or physbo.variable
             The set of candidates. Each row vector represents the feature vector of each search candidate.
        config: set_config object (physbo.misc.set_config)
        """
        self.predictor = None
        self.training = variable()
        self.test = self._set_test(test_X)
        self.actions = np.arange(0, self.test.X.shape[0])
        self.history = history()
        self.config = self._set_config(config)

    def set_seed(self, seed):
        """
        Setting a seed parameter for np.random.

        Parameters
        ----------
        seed: int
            seed number
        -------

        """
        self.seed = seed
        np.random.seed(self.seed)

    def delete_actions(self, index, actions=None):
        """
        Deleteing actions

        Parameters
        ----------
        index: int
            Index of an action to be deleted.
        actions: numpy.ndarray
            Array of actions.
        Returns
        -------
        actions: numpy.ndarray
            Array of actions which does not include action specified by index.
        """
        actions = self._set_unchosen_actions(actions)
        return np.delete(actions, index)

    def write(self, action, t, X=None):
        """
        Writing history (update history, not output to a file).

        Parameters
        ----------
        action: numpy.ndarray
            Indexes of actions.
        t:  numpy.ndarray
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        X:  numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.

        Returns
        -------

        """
        if X is None:
            X = self.test.X[action, :]
            Z = self.test.Z[action, :] if self.test.Z is not None else None
        else:
            Z = self.predictor.get_basis(X) if self.predictor is not None else None

        self.new_data = variable(X, t, Z)
        self.history.write(t, action)
        self.training.add(X=X, t=t, Z=Z)

    def random_search(
        self, max_num_probes, num_search_each_probe=1, simulator=None, is_disp=True
    ):
        """
        Performing random search.

        Parameters
        ----------
        max_num_probes: int
            Maximum number of random search process.
        num_search_each_probe: int
            Number of search at each random search process.
        simulator: simulator object
            This object is called in call_simulator and must have __call__(action).
            Here, action is an integer which represents the index of the candidate.
        is_disp: bool
            If true, process messages are outputted.
        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """

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
                utility.show_start_message_multi_search(self.history.num_runs)

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def bayes_search(
        self,
        training=None,
        max_num_probes=None,
        num_search_each_probe=1,
        predictor=None,
        is_disp=True,
        simulator=None,
        score="TS",
        interval=0,
        num_rand_basis=0,
    ):
        """
        Performing Bayesian optimization.

        Parameters
        ----------
        training: physbo.variable
            Training dataset.
        max_num_probes: int
            Maximum number of searching process by Bayesian optimization.
        num_search_each_probe: int
            Number of searching by Bayesian optimization at each process.
        predictor: predictor object
            Base class is defined in physbo.predictor.
            If None, blm_predictor is defined.
        is_disp: bool
             If true, process messages are outputted.
        simulator: simulator object
            This object is called in call_simulator and must have __call__(action).
            Here, action is an integer which represents the index of the candidate.
        score: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
        interval: int
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
        num_rand_basis: int
            The number of basis function. If you choose 0, ordinary Gaussian process run.

        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        self.training = self._set_training(training)

        if predictor is None:
            self.predictor = self._init_predictor(is_rand_expans)
        else:
            self.predictor = predictor

        N = int(num_search_each_probe)

        for n in range(max_num_probes):

            if utility.is_learning(n, interval):
                self.predictor.fit(self.training, num_rand_basis)
                self.test.Z = self.predictor.get_basis(self.test.X)
                self.training.Z = self.predictor.get_basis(self.training.X)
                self.predictor.prepare(self.training)
            else:
                try:
                    self.predictor.update(self.training, self.new_data)
                except:
                    self.predictor.prepare(self.training)

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search(self.history.num_runs, score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self.get_actions(score, N, K, alpha)

            if simulator is None:
                return action

            t, X = call_simulator(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results(self.history, N)

        return copy.deepcopy(self.history)

    def get_score(self, mode, predictor=None, training=None, alpha=1):
        """
        Getting score.

        Parameters
        ----------
        mode: str
            The type of aquision funciton. TS, EI and PI are available.
            These functions are defined in score.py.
        predictor: predictor object
            Base class is defined in physbo.predictor.
        training:physbo.variable
            Training dataset.
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).
        Returns
        -------
        f: float or list of float
            Score defined in each mode.
        """
        self._set_training(training)
        self._set_predictor(predictor)
        actions = self.actions

        test = self.test.get_subset(actions)
        if mode == "EI":
            f = physbo.search.score.EI(predictor, training, test)
        elif mode == "PI":
            f = physbo.search.score.PI(predictor, training, test)
        elif mode == "TS":
            f = physbo.search.score.TS(predictor, training, test, alpha)
        else:
            raise NotImplementedError("mode must be EI, PI or TS.")
        return f

    def get_marginal_score(self, mode, chosen_actions, N, alpha):
        """
        Getting marginal scores.

        Parameters
        ----------
        mode: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        chosen_actions: numpy.ndarray
            Array of selected actions.
        N: int
            The total number of search candidates.
        alpha: float
            not used.

        Returns
        -------
        f: list
            N dimensional scores (score is defined in each mode)
        """
        f = np.zeros((N, len(self.actions)))
        new_test = self.test.get_subset(chosen_actions)
        virtual_t = self.predictor.get_predict_samples(self.training, new_test, N)

        for n in range(N):
            predictor = copy.deepcopy(self.predictor)
            train = copy.deepcopy(self.training)
            virtual_train = new_test
            virtual_train.t = virtual_t[n, :]

            if virtual_train.Z is None:
                train.add(virtual_train.X, virtual_train.t)
            else:
                train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            try:
                predictor.update(train, virtual_train)
            except:
                predictor.prepare(train)

            f[n, :] = self.get_score(mode, predictor, train)
        return f

    def get_actions(self, mode, N, K, alpha):
        """
        Getting actions

        Parameters
        ----------
        mode: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        N: int
            The total number of selected candidates.
        K: int
            The total number of search candidates.
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).

        Returns
        -------
        chosen_actions: numpy.ndarray
            An N-dimensional array of actions selected in each search process.
        """
        f = self.get_score(mode, self.predictor, self.training, alpha)
        temp = np.argmax(f)
        action = self.actions[temp]
        self.actions = self.delete_actions(temp)

        chosen_actions = np.zeros(N, dtype=int)
        chosen_actions[0] = action

        for n in range(1, N):
            f = self.get_marginal_score(mode, chosen_actions[0:n], K, alpha)
            temp = np.argmax(np.mean(f, 0))
            chosen_actions[n] = self.actions[temp]
            self.actions = self.delete_actions(temp)

        return chosen_actions

    def get_random_action(self, N):
        """
        Getting indexes of actions randomly.

        Parameters
        ----------
        N: int
            Total number of search candidates.
        Returns
        -------
        action: numpy.ndarray
            Indexes of actions selected randomly from search candidates.
        """
        random_index = np.random.permutation(range(self.actions.shape[0]))
        index = random_index[0:N]
        action = self.actions[index]
        self.actions = self.delete_actions(index)
        return action

    def save(self, file_history, file_training=None, file_predictor=None):
        """

        Saving history, training and predictor into the corresponding files.

        Parameters
        ----------
        file_history: str
            The name of the file that stores the information of the history.
        file_training: str
            The name of the file that stores the training dataset.
        file_predictor: str
            The name of the file that stores the predictor dataset.

        Returns
        -------

        """
        self.history.save(file_history)

        if file_training is not None:
            self.training.save(file_training)

        if file_predictor is not None:
            with open(file_predictor, "wb") as f:
                pickle.dump(self.predictor, f)

    def load(self, file_history, file_training=None, file_predictor=None):
        """

        Loading files about history, training and predictor.

        Parameters
        ----------
        file_history: str
            The name of the file that stores the information of the history.
        file_training: str
            The name of the file that stores the training dataset.
        file_predictor: str
            The name of the file that stores the predictor dataset.

        Returns
        -------

        """
        self.history.load(file_history)

        if file_training is None:
            N = self.history.total_num_search
            X = self.test.X[self.history.chosen_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training = variable(X=X, t=t)
        else:
            self.training = variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, "rb") as f:
                self.predictor = pickle.load(f)

    def export_predictor(self):
        """
        Returning the predictor dataset

        Returns
        -------

        """
        return self.predictor

    def export_training(self):
        """
        Returning the training dataset

        Returns
        -------

        """
        return self.training

    def export_history(self):
        """
        Returning the information of the history.

        Returns
        -------

        """
        return self.history

    def _set_predictor(self, predictor=None):
        """

        Set predictor if defined.

        Parameters
        ----------
        predictor: predictor object
            Base class is defined in physbo.predictor.

        Returns
        -------

        """
        if predictor is None:
            predictor = self.predictor
        return predictor

    def _init_predictor(self, is_rand_expans, predictor=None):
        """
        Setting the initial predictor.

        Parameters
        ----------
        is_rand_expans: bool
        If true, blm_predictor is selected.
        If false, gp_predictor is selected.
        predictor: predictor object
            Base class is defined in physbo.predictor.

        Returns
        -------
        predictor: predictor object
            Base class is defined in physbo.predictor.
        """
        self.predictor = self._set_predictor(predictor)
        if self.predictor is None:
            if is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)

        return self.predictor

    def _set_training(self, training=None):
        """

        Set training dataset.

        Parameters
        ----------
        training: physbo.variable
            Training dataset.

        Returns
        -------
        training: physbo.variable
            Training dataset.
        """
        if training is None:
            training = self.training
        return training

    def _set_unchosen_actions(self, actions=None):
        """

        Parameters
        ----------
        actions: numpy.ndarray
            An array of indexes of the actions which are not chosen.

        Returns
        -------
         actions: numpy.ndarray
            An array of indexes of the actions which are not chosen.

        """
        if actions is None:
            actions = self.actions
        return actions

    def _set_test(self, test_X):
        """
        Set test candidates.

        Parameters
        ----------
        test_X: numpy.ndarray or physbo.variable
             The set of candidates. Each row vector represents the feature vector of each search candidate.
        Returns
        -------
        test_X: numpy.ndarray or physbo.variable
             The set of candidates. Each row vector represents the feature vector of each search candidate.
        """
        if isinstance(test_X, np.ndarray):
            test = variable(X=test_X)
        elif isinstance(test_X, variable):
            test = test_X
        else:
            raise TypeError(
                "The type of test_X must \
                             take ndarray or physbo.variable"
            )
        return test

    def _set_config(self, config=None):
        """
        Set configure information.

        Parameters
        ----------
        config: set_config object (physbo.misc.set_config)

        Returns
        -------
        config: set_config object (physbo.misc.set_config)

        """
        if config is None:
            config = physbo.misc.set_config()
        return config


class policy_mo(policy):
    def __init__(self, test_X, num_objectives, config=None, initial_actions=None):
        self.num_objectives = num_objectives
        self.history = history_mo(num_objectives=self.num_objectives)

        self.training_list = [variable() for i in range(self.num_objectives)]
        self.predictor_list = [None for i in range(self.num_objectives)]
        self.test_list = [self._set_test(test_X) for i in range(self.num_objectives)]
        self.new_data_list = [None for i in range(self.num_objectives)]

        self.actions = np.arange(0, test_X.shape[0])
        self.config = self._set_config(config)

        self.TS_candidate_num = None

        if initial_actions is not None:
            self.actions = sorted(list(set(self.actions)-set(initial_actions)))


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
                Z = predictor.get_basis(X) \
                    if predictor is not None else None

            self.new_data_list[i] = variable(X, t[:, i], Z)
            self.training_list[i].add(X=X, t=t[:, i], Z=Z)

    def _switch_model(self, i):
        self.training = self.training_list[i]
        self.predictor = self.predictor_list[i]
        self.test = self.test_list[i]
        self.new_data = self.new_data_list[i]

    def random_search(self, max_num_probes, num_search_each_probe=1,
                      simulator=None, is_disp=True, disp_pareto_set=False):

        N = int(num_search_each_probe)

        if int(max_num_probes) * N > len(self.actions):
            raise ValueError('max_num_probes * num_search_each_probe must \
                be smaller than the length of candidates')

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            if is_disp and N > 1:
                utility.show_start_message_multi_search_mo(self.history.num_runs, "random")

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t, X = call_simulator_mo(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results_mo(self.history, N, disp_pareto_set=disp_pareto_set)

        return copy.deepcopy(self.history)

    def bayes_search(self, training_list=None, max_num_probes=None,
                     num_search_each_probe=1,
                     predictor_list=None, is_disp=True, disp_pareto_set=False,
                     simulator=None, score='HVPI', interval=0,
                     num_rand_basis=0):

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        if training_list is not None:
            self.training_list = training_list

        if predictor_list is None:
            if is_rand_expans:
                self.predictor_list = [blm_predictor(self.config) for i in range(self.num_objectives)]
            else:
                self.predictor_list = [gp_predictor(self.config) for i in range(self.num_objectives)]
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
                try:
                    for i in range(self.num_objectives):
                        self._switch_model(i)
                        self.predictor.update(self.training, self.new_data)
                except:
                    for i in range(self.num_objectives):
                        self._switch_model(i)
                        self.predictor.prepare(self.training)

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search_mo(self.history.num_runs, score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self.get_actions(score, N, K, alpha)

            if simulator is None:
                return action

            t, X = call_simulator_mo(simulator, action)

            self.write(action, t, X)

            if is_disp:
                utility.show_search_results_mo(self.history, N, disp_pareto_set=disp_pareto_set)

        return copy.deepcopy(self.history)

    def get_actions(self, mode, N, K, alpha):
        f = self.get_score(mode, self.predictor_list, self.training_list, self.history.pareto, alpha)
        temp = np.argmax(f)
        action = self.actions[temp]
        self.actions = self.delete_actions(temp)

        chosed_actions = np.zeros(N, dtype=int)
        chosed_actions[0] = action

        for n in range(1, N):
            f = self.get_score(mode, self.predictor_list, self.training_list, self.history.pareto, alpha)
            temp = np.argmax(np.mean(f, 0))
            chosed_actions[n] = self.actions[temp]
            self.actions = self.delete_actions(temp)

        return chosed_actions

    def get_score(self, mode, predictor_list, training_list, pareto, alpha=1):
        actions = self.actions
        test = self.test.get_subset(actions)

        if mode == 'EHVI':
            fmean, fstd = self.get_fmean_fstd(predictor_list, training_list, test)
            f = physbo.search.score.EHVI(fmean, fstd, pareto)
        elif mode == 'HVPI':
            fmean, fstd = self.get_fmean_fstd(predictor_list, training_list, test)
            f = physbo.search.score.HVPI(fmean, fstd, pareto)
        elif mode == 'TS':
            f = physbo.search.score.TS_MO(predictor_list, training_list, test, alpha,
                                         reduced_candidate_num=self.TS_candidate_num)
        else:
            raise NotImplementedError('mode must be EHVI, HVPI or TS.')
        return f

    def get_fmean_fstd(self, predictor_list, training_list, test):
        fmean = [predictor.get_post_fmean(training, test) \
                 for predictor, training in zip(predictor_list, training_list)]
        fcov = [predictor.get_post_fcov(training, test) \
                for predictor, training in zip(predictor_list, training_list)]

        # shape: (N, n_obj)
        fmean = np.array(fmean).T
        fstd = np.sqrt(np.array(fcov)).T
        return fmean, fstd

    def load(self, file_history, file_training_list=None, file_predictor_list=None):
        self.history.load(file_history)

        if file_training_list is None:
            N = self.history.total_num_search
            X = self.test_list[0].X[self.history.chosed_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training_list = [variable(X=X, t=t[:, i]) for i in range(self.num_objectives)]
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

    def save_predictor_list(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.predictor_list, f, 2)

    def save_training_list(self, file_name):
        obj = [{"X": training.X, "t": training.t, "Z": training.Z} for training in self.training_list]
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, 2)

    def load_predictor_list(self, file_name):
        with open(file_name, 'rb') as f:
            self.predictor_list = pickle.load(f)

    def load_training_list(self, file_name):
        with open(file_name, 'rb') as f:
            data_list = pickle.load(f)

        self.training_list = [variable() for i in range(self.num_objectives)]
        for data, training in zip(data_list, self.training_list):
            training.X = data['X']
            training.t = data['t']
            training.Z = data['Z']
