import numpy as np
import copy
import pickle as pickle
import itertools
import time

from .results import history
from .. import utility
from .. import score as search_score
from ...gp import predictor as gp_predictor
from ...blm import predictor as blm_predictor
from ...misc import set_config

from physbo.variable import variable


class policy:
    def __init__(self, test_X, config=None, initial_data=None, comm=None):
        """

        Parameters
        ----------
        test_X: numpy.ndarray or physbo.variable
             The set of candidates. Each row vector represents the feature vector of each search candidate.
        config: set_config object (physbo.misc.set_config)
        initial_data: tuple[np.ndarray, np.ndarray]
            The initial training datasets.
            The first elements is the array of actions and the second is the array of value of objective functions
        comm: MPI.Comm, optional
            MPI Communicator
        """
        self.predictor = None
        self.training = variable()
        self.new_data = None
        self.test = self._make_variable_X(test_X)
        self.actions = np.arange(0, self.test.X.shape[0])
        self.history = history()
        if config is None:
            self.config = set_config()
        else:
            self.config = config

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
                raise RuntimeError(msg)
            actions, fs = initial_data
            if len(actions) != len(fs):
                msg = "ERROR: len(initial_data[0]) != len(initial_data[1])"
                raise RuntimeError(msg)
            self.write(actions, fs)
            self.actions = np.array(sorted(list(set(self.actions) - set(actions))))

        if comm is None:
            self.mpicomm = None
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpicomm = comm
            self.mpisize = comm.size
            self.mpirank = comm.rank
            self.actions = np.array_split(self.actions, self.mpisize)[self.mpirank]
            self.config.learning.is_disp = (self.config.learning.is_disp and self.mpirank == 0)

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

    def write(
        self,
        action,
        t,
        X=None,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
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
        if X is None:
            X = self.test.X[action, :]
            Z = self.test.Z[action, :] if self.test.Z is not None else None
        else:
            Z = self.predictor.get_basis(X) if self.predictor is not None else None

        self.history.write(
            t,
            action,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )
        self.training.add(X=X, t=t, Z=Z)
        if self.new_data is None:
            self.new_data = variable(X=X, t=t, Z=Z)
        else:
            self.new_data.add(X=X, t=t, Z=Z)

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
        simulator: callable
            Callable (function or object with ``__call__``) from action to t
            Here, action is an integer which represents the index of the candidate.
        is_disp: bool
            If true, process messages are outputted.
        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """

        if self.mpirank != 0:
            is_disp = False

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            time_total = time.time()
            if is_disp and N > 1:
                utility.show_start_message_multi_search(self.history.num_runs)

            time_get_action = time.time()
            action = self._get_random_action(N)
            time_get_action = time.time() - time_get_action

            N_indeed = len(action)
            if N_indeed == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                return copy.deepcopy(self.history)

            if simulator is None:
                return action

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action,
                t,
                time_total=[time_total] * N_indeed,
                time_update_predictor=np.zeros(N_indeed, dtype=float),
                time_get_action=[time_get_action] * N_indeed,
                time_run_simulator=[time_run_simulator] * N_indeed,
            )

            if is_disp:
                utility.show_search_results(self.history, N_indeed)

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
        simulator: callable
            Callable (function or object with ``__call__``)
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

        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = num_rand_basis != 0

        if training is not None:
            self.training = training

        if predictor is not None:
            self.predictor = predictor
        elif self.predictor is None:
            self._init_predictor(is_rand_expans)

        if max_num_probes == 0 and interval >= 0:
            self._learn_hyperparameter(num_rand_basis)

        N = int(num_search_each_probe)

        for n in range(max_num_probes):
            time_total = time.time()

            time_update_predictor = time.time()
            if utility.is_learning(n, interval):
                self._learn_hyperparameter(num_rand_basis)
            else:
                self._update_predictor()
            time_update_predictor = time.time() - time_update_predictor

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search(self.history.num_runs, score)

            time_get_action = time.time()
            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self._get_actions(score, N, K, alpha)
            time_get_action = time.time() - time_get_action

            N_indeed = len(action)
            if N_indeed == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                break

            if simulator is None:
                self.config.learning.is_disp = old_disp
                return action

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action,
                t,
                time_total=[time_total] * N_indeed,
                time_update_predictor=[time_update_predictor] * N_indeed,
                time_get_action=[time_get_action] * N_indeed,
                time_run_simulator=[time_run_simulator] * N_indeed,
            )

            if is_disp:
                utility.show_search_results(self.history, N_indeed)
        self._update_predictor()
        self.config.learning.is_disp = old_disp
        return copy.deepcopy(self.history)

    @staticmethod
    def _warn_no_predictor(method_name):
        print("Warning: Since policy.predictor is not yet set,")
        print("         a GP predictor (num_rand_basis=0) is used for predicting")
        print("         If you want to use a BLM predictor (num_rand_basis>0),")
        print("         call bayes_search(max_num_probes=0, num_rand_basis=nrb)")
        print("         before calling {}.".format(method_name))

    def get_post_fmean(self, xs):
        """Calculate mean value of predictor (post distribution)"""
        X = self._make_variable_X(xs)
        predictor = self.predictor
        if predictor is None:
            self._warn_no_predictor("get_post_fmean()")
            predictor = gp_predictor(self.config)
            predictor.fit(self.training, 0)
            predictor.prepare(self.training)
        return predictor.get_post_fmean(self.training, X)

    def get_post_fcov(self, xs):
        """Calculate covariance of predictor (post distribution)"""
        X = self._make_variable_X(xs)
        predictor = self.predictor
        if predictor is None:
            self._warn_no_predictor("get_post_fcov()")
            predictor = gp_predictor(self.config)
            predictor.fit(self.training, 0)
            predictor.prepare(self.training)
        return predictor.get_post_fcov(self.training, X)

    def get_score(
        self,
        mode,
        *,
        actions=None,
        xs=None,
        predictor=None,
        training=None,
        parallel=True,
        alpha=1
    ):
        """
        Calcualte score (acquisition function)

        Parameters
        ----------
        mode: str
            The type of aquisition funciton. TS, EI and PI are available.
            These functions are defined in score.py.
        actions: array of int
            actions to calculate score
        xs: physbo.variable or np.ndarray
            input parameters to calculate score
        predictor: predictor object
            predictor used to calculate score.
            If not given, self.predictor will be used.
        training:physbo.variable
            Training dataset.
            If not given, self.training will be used.
        parallel: bool
            Calculate scores in parallel by MPI (default: True)
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).

        Returns
        -------
        f: float or list of float
            Score defined in each mode.

        Raises
        ------
        RuntimeError
            If both *actions* and *xs* are given

        Notes
        -----
        When neither *actions* nor *xs* are given, scores for actions not yet searched will be calculated.

        When *parallel* is True, it is assumed that the function receives the same input (*actions* or *xs*) for all the ranks.
        If you want to split the input array itself, set *parallel* be False and merge results by yourself.
        """
        if training is None:
            training = self.training

        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor is None:
            predictor = self.predictor

        if predictor is None:
            self._warn_no_predictor("get_score()")
            predictor = gp_predictor(self.config)
            predictor.fit(training, 0)
            predictor.prepare(training)

        if xs is not None:
            if actions is not None:
                raise RuntimeError("ERROR: both actions and xs are given")
            test = self._make_variable_X(xs)
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
            test = self.test.get_subset(actions)

        f = search_score.score(
            mode, predictor=predictor, training=training, test=test, alpha=alpha
        )
        if parallel and self.mpisize > 1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f

    def _get_marginal_score(self, mode, chosen_actions, K, alpha):
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
        K: int
            The number of samples for evaluating score.
        alpha: float
            not used.

        Returns
        -------
        f: list
            N dimensional scores (score is defined in each mode)
        """
        f = np.zeros((K, len(self.actions)), dtype=float)

        # draw K samples of the values of objective function of chosen actions
        new_test_local = self.test.get_subset(chosen_actions)
        virtual_t_local = self.predictor.get_predict_samples(self.training, new_test_local, K)
        if self.mpisize == 1:
            new_test = new_test_local
            virtual_t = virtual_t_local
        else:
            new_test = variable()
            for nt in self.mpicomm.allgather(new_test_local):
                new_test.add(X=nt.X, t=nt.t, Z=nt.Z)
            virtual_t = np.concatenate(self.mpicomm.allgather(virtual_t_local), axis=1)
        # virtual_t = self.predictor.get_predict_samples(self.training, new_test, K)

        for k in range(K):
            predictor = copy.deepcopy(self.predictor)
            train = copy.deepcopy(self.training)
            virtual_train = new_test
            virtual_train.t = virtual_t[k, :]

            if virtual_train.Z is None:
                train.add(virtual_train.X, virtual_train.t)
            else:
                train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            predictor.update(train, virtual_train)

            f[k, :] = self.get_score(
                mode, predictor=predictor, training=train, parallel=False
            )
        return np.mean(f, axis=0)

    def _get_actions(self, mode, N, K, alpha):
        """
        Getting next candidates

        Parameters
        ----------
        mode: str
            The type of aquisition funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        N: int
            The total number of actions to return.
        K: int
            The total number of samples to evaluate marginal score
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).

        Returns
        -------
        chosen_actions: numpy.ndarray
            An N-dimensional array of actions selected in each search process.
        """
        f = self.get_score(
            mode,
            predictor=self.predictor,
            training=self.training,
            alpha=alpha,
            parallel=False,
        )
        champion, local_champion, local_index = self._find_champion(f)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f = self._get_marginal_score(mode, chosen_actions[0:n], K, alpha)
            champion, local_champion, local_index = self._find_champion(f)
            if champion == -1:
                break
            if champion == local_champion:
                self.actions = self._delete_actions(local_index)
            chosen_actions.append(champion)
        return np.array(chosen_actions)

    def _find_champion(self, f):
        if len(f) == 0:
            local_fmax = -float("inf")
            local_index = -1
            local_champion = -1
        else:
            local_fmax = np.max(f)
            local_index = np.argmax(f)
            local_champion = self.actions[local_index]
        if self.mpisize == 1:
            champion = local_champion
        else:
            local_champions = self.mpicomm.allgather(local_champion)
            local_fs = self.mpicomm.allgather(local_fmax)
            champion_rank = np.argmax(local_fs)
            champion = local_champions[champion_rank]
        return champion, local_champion, local_index

    def _get_random_action(self, N):
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
        if self.mpisize == 1:
            n = len(self.actions)
            if n <= N:
                index = np.arange(0, n)
            else:
                index = np.random.choice(len(self.actions), N, replace=False)
            action = self.actions[index]
            self.actions = self._delete_actions(index)
        else:
            nactions = self.mpicomm.gather(len(self.actions), root=0)
            local_indices = [[] for _ in range(self.mpisize)]
            if self.mpirank == 0:
                hi = np.add.accumulate(nactions)
                lo = np.roll(hi, 1)
                lo[0] = 0
                if hi[-1] <= N:
                    index = np.arange(0, hi[-1])
                else:
                    index = np.random.choice(hi[-1], N, replace=False)
                ranks = np.searchsorted(hi, index, side="right")
                for r, i in zip(ranks, index):
                    local_indices[r].append(i - lo[r])
            local_indices = self.mpicomm.scatter(local_indices, root=0)
            local_actions = self.actions[local_indices]
            self.actions = self._delete_actions(local_indices)
            action = self.mpicomm.allgather(local_actions)
            action = itertools.chain.from_iterable(action)
            action = np.array(list(action))
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
        if self.mpirank == 0:
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

    def _init_predictor(self, is_rand_expans):
        """
        Initialize predictor.

        Parameters
        ----------
        is_rand_expans: bool
            If true, physbo.blm.predictor is selected.
            If false, physbo.gp.predictor is selected.
        """
        if is_rand_expans:
            self.predictor = blm_predictor(self.config)
        else:
            self.predictor = gp_predictor(self.config)

    def _learn_hyperparameter(self, num_rand_basis):
        self.predictor.fit(self.training, num_rand_basis)
        self.test.Z = self.predictor.get_basis(self.test.X)
        self.training.Z = self.predictor.get_basis(self.training.X)
        self.predictor.prepare(self.training)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            self.predictor.update(self.training, self.new_data)
            self.new_data = None

    def _make_variable_X(self, test_X):
        """
        Make a new *variable* with X=test_X

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
            raise TypeError("The type of test_X must be ndarray or physbo.variable")
        return test

    def _delete_actions(self, index, actions=None):
        """
        Returns remaining actions

        Notes
        -----
        This method itself does not modify *self*

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
        if actions is None:
            actions = self.actions
        return np.delete(actions, index)


def _run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)
