# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
import numpy as np


def default_alg_dict(min_X: np.ndarray, max_X: np.ndarray, algorithm_name: str = "exchange"):
    """
    Return the default algorithm parameter settings for the given algorithm name.

    Parameters
    ----------
    algorithm_name : str, optional (default: "exchange")
        The name of the algorithm to use.
        "exchange": Exchange Monte Carlo (default)
        "pamc": Population Annealing Monte Carlo
        "minsearch": Nelder-Mead method
        "mapper": Grid search
        "bayes": Bayesian optimization

    Returns
    -------
    dict
        The default algorithm parameter settings for the given algorithm name.
    """

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
            }
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
            "minimize": {"maxiter": 100}
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
            "bayes": {"random_max_num_probes": 10, "bayes_max_num_probes": 40, "score": "EI", "num_rand_basis": 0}
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def optimize(fn, min_X, max_X, alg_dict=None, mpicomm=None):
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
    alg_dict : dict, optional
        Dictionary specifying the optimization algorithm and its parameters.
        If None, uses default parameters from default_alg_dict().
    mpicomm : MPI.Comm, optional
        MPI communicator for parallel optimization. If None, runs in serial.

    Returns
    -------
    numpy.ndarray
        The optimal point found by the optimizer
    """

    import odatse
    import odatse.solver.function

    dim = len(min_X)
    L_X = max_X - min_X
    d_X = L_X / 100

    if alg_dict is None:
        alg_dict = default_alg_dict(min_X, max_X)

    base_dict = {
        "dimension": dim,
        "root_dir": ".",
        "output_dir": "odatse_output",
    }

    info_dict = {"base": base_dict, "algorithm": alg_dict, "solver": {}}

    info = odatse.Info(info_dict)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(fn)
    runner = odatse.Runner(solver, info)
    alg_module = odatse.algorithm.choose_algorithm(alg_dict["name"])

    try:
        # try development version of odatse
        alg = alg_module.Algorithm(info, runner, mpicomm=mpicomm)
    except TypeError:
        alg = alg_module.Algorithm(info, runner)

    result = alg.main()

    if alg_dict.get("name") == "mapper":
        with open("odatse_output/ColorMap.txt") as f:
            X = np.zeros((1,dim))
            fx = np.inf
            for line in f:
                words = line.split()
                if float(words[-1]) < fx:
                    fx = float(words[-1])
                    X = np.array([float(x) for x in words[0:-1]])
    else:
        X = result["x"]

    minsearch_after_optimizer = False

    if minsearch_after_optimizer:
        alg_dict_minsearch = copy.deepcopy(alg_dict)
        alg_dict_minsearch["name"] = "minsearch"
        alg_dict_minsearch["param"] = {}
        alg_dict_minsearch["param"]["min_list"] = min_X
        alg_dict_minsearch["param"]["max_list"] = max_X
        alg_dict_minsearch["param"]["unit_list"] = d_X
        alg_dict_minsearch["param"]["initial_list"] = X
        alg_dict_minsearch["minimize"] = {"initial_scale_list": -0.01*X, "maxiter": 100}

        info_dict_minsearch = {"base": base_dict, "algorithm": alg_dict_minsearch, "solver": {}}
        info.from_dict(info_dict_minsearch)
        # solver = odatse.solver.function.Solver(info)
        # solver.set_function(fn)
        # runner = odatse.Runner(solver, info)
        alg_module = odatse.algorithm.choose_algorithm("minsearch")
        try:
            # try development version of odatse
            alg = alg_module.Algorithm(info, runner, mpicomm=mpicomm)
        except TypeError:
            alg = alg_module.Algorithm(info, runner)

        result = alg.main()

        return result["x"].reshape(1, -1)
    else:
        return X.reshape(1, -1)
