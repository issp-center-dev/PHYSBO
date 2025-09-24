# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
import numpy as np


import odatse
import odatse.solver.function


def default_alg_dict(
    min_X: np.ndarray, max_X: np.ndarray, algorithm_name: str = "mapper"
):
    """
    Return the default algorithm parameter settings for the given algorithm name.

    Parameters
    ----------
    algorithm_name : str, optional (default: "mapper")
        The name of the algorithm to use.
        "exchange": Exchange Monte Carlo
        "pamc": Population Annealing Monte Carlo
        "minsearch": Nelder-Mead method
        "mapper": Grid search
        "bayes": Bayesian optimization

    Returns
    -------
    dict
        The default algorithm parameter settings for the given algorithm name.
    """

    min_X = np.array(min_X)
    max_X = np.array(max_X)
    dim = len(min_X)
    L_X = np.array(max_X) - np.array(min_X)
    d_X = L_X / 100

    if algorithm_name == "exchange":
        return {
            "name": "exchange",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "step_list": d_X,
            },
            "exchange": {
                "numsteps": 1000,
                "numsteps_exchange": 10,
                "Tmin": 0.1,
                "Tmax": 10.0,
                "Tlogspace": True,
                "nreplica_per_proc": 10,
            },
        }
    elif algorithm_name == "pamc":
        return {
            "name": "pamc",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "step_list": d_X,
            },
            "pamc": {
                "numsteps_annealing": 10,
                "Tnum": 11,
                "Tmax": 10.0,
                "Tmin": 0.1,
                "Tlogspace": True,
                "nreplica_per_proc": 100,
                "resampling_interval": 2,
                "fix_num_replicas": True,
            },
        }
    elif algorithm_name == "minsearch":
        return {
            "name": "minsearch",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "unit_list": d_X,
            },
            "minimize": {"maxiter": 100},
        }
    elif algorithm_name == "mapper":
        return {
            "name": "mapper",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "num_list": 11 * np.ones(dim, dtype=int),
            },
        }
    elif algorithm_name == "bayes":
        return {
            "name": "bayes",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "num_list": 21 * np.ones(dim, dtype=int),
            },
            "bayes": {
                "random_max_num_probes": 10,
                "bayes_max_num_probes": 40,
                "score": "EI",
                "num_rand_basis": 0,
            },
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


class Optimizer:
    def __init__(self, alg_dict):
        """
        Parameters
        ----------
        alg_dict : dict
            Dictionary specifying the optimization algorithm and its parameters.
        """
        self.alg_dict = copy.deepcopy(alg_dict)

    def __call__(self, fn, mpicomm=None):
        """
        Optimize a function using ODATSE (Open-source Data Assimilation and Topological Structure Exploration)

        Parameters
        ----------
        fn : callable
            The objective function to minimize. Should take a numpy array of shape (dim,) and return a float.
        min_X : numpy.ndarray
            Lower bounds for each dimension
        max_X : numpy.ndarray
            Upper bounds for each dimension
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel optimization. If None, runs in serial.

        Returns
        -------
        numpy.ndarray
            The optimal point found by the optimizer
        """

        dim = self.alg_dict["param"]["min_list"].size

        base_dict = {
            "dimension": dim,
            "root_dir": ".",
            "output_dir": "odatse_output",
        }

        info_dict = {"base": base_dict, "algorithm": self.alg_dict, "solver": {}}

        info = odatse.Info(info_dict)
        solver = odatse.solver.function.Solver(info)
        solver.set_function(
            lambda x: -fn(x)
        )  # ODATSE minimizes the function, so we need to negate the objective function
        runner = odatse.Runner(solver, info)
        alg_module = odatse.algorithm.choose_algorithm(self.alg_dict["name"])

        try:
            # try development version of odatse
            alg = alg_module.Algorithm(info, runner, mpicomm=mpicomm)
        except TypeError:
            alg = alg_module.Algorithm(info, runner)

        result = alg.main()

        if self.alg_dict.get("name") == "mapper":
            with open("odatse_output/ColorMap.txt") as f:
                X = np.zeros((1, dim))
                fx = np.inf
                for line in f:
                    words = line.split()
                    if float(words[-1]) < fx:
                        fx = float(words[-1])
                        X = np.array([float(x) for x in words[0:-1]])
        else:
            X = result["x"]

        return X.reshape(1, -1)
