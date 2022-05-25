Basic usage
=====================

Install
---------------------

Required Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Python >= 3.6
* numpy
* scipy

Download and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- From ``PyPI`` (recommended) ::

  $ pip3 install physbo

  - Required packages such as NumPy will also be installed at the same time.

  - If you add the ``--user`` option, it will be installed under the user's home directory ::

    $ pip3 install --user physbo


- From source (for developers)

  #. Download or clone the github repository
          
       $ git clone https://github.com/issp-center-dev/PHYSBO

  #. Update ``pip`` to 19.0 or higher ::

       $ pip3 install -U pip

       - If you don't have ``pip3``, you can install it with ``python3 -m ensurepip``.

  #. Install ::

       $ cd PHYSBO
       $ pip3 install --user ./

Uninstall
~~~~~~~~~~~~~~~~~~~~~~~~

#. Execute the following command. ::

   $ pip uninstall physbo


Basic structures
--------------------------

PHYSBO has the following structure (shown up to the second level).

..
 |--physbo
 |    |--blm
 |    |--gp
 |    |--misc
 |    |--opt
 |    |--search
 |    |--predictor.py
 |    |--variable.py

Each module is created with the following structure.
 
- ``blm`` :Module for Baysean linear model
- ``gp`` :Module for Gaussian Process
- ``opt`` :Module for optimazation
- ``search`` :Module for searching for optimal solutions
- ``predictor.py`` :Abstract class for predictors
- ``variable.py`` :Class defined for variable associations used in physbo
- ``misc`` : Others (e.g., modules for normalizing the search space)

For more information about each module, please refer to the API reference.
 
Calculation flow
--------------------------

Bayesian optimization is well suited for optimization problems such as complex simulations or real-world experimental tasks where the objective function is very costly to evaluate.
In PHYSBO, the following steps are used to perform the optimization (please refer to the tutorial and API reference for details on each).

1. Defining the search space

 Define each parameter set (d-dimensional vector) as a search candidate, where N: the number of search candidates , d: the number of input parameter dimensions. The parameter set should list all the candidates.

2. Defining the simulator

 For searching candidates defined above, define a simulator that gives the objective function values (values to be optimized, such as material property values) for each search candidate. In PHYSBO, the direction of optimization is to maximize the objective function, so if you want to minimize the objective function, you can do so by applying a negative value to the value returned by the simulator.
    
3. Performing optimization

 First, set the optimization policy (the search space is passed to policy as an argument at this stage). You can choose between the following two optimization methods.
  
  - ``random_search``
  - ``bayes_search``

  In ``random_search``, we randomly select parameters from the search space and search for the largest objective function among them. It is used to prepare an initial set of parameters as a preprocessing step for Bayesian optimization. ``bayes_search`` performs Bayesian optimization. The type of score (acquisition function) in Bayesian optimization can be one of the following.

  - TS (Thompson Sampling): Sample one regression function from the posterior probability distribution of the learned Gaussian process, and select the point where the predicetd value becomes maximum as a next candidate.
  - EI (Expected Improvement): Select the point where the expected value of the difference between the predicted value by the Gaussian process and the maximum value in the current situation becomes the maximum as a next candidate.
  - PI (Probability of Improvement): Select the point with the highest probability of exceeding the current maximum of the current acquisition function as a next candidate.
  
  Details of  Gaussian processes are described in :ref:`chap_algorithm` . For other details of each method, please see  `this reference <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ .
  If you specify the simulator and the number of search steps in these methods, the following loop will rotate by the number of search steps.

    i). Select the next parameter to be executed from the list of candidate parameters.
    
    ii). Run the simulator with the selected parameters.

  The number of parameter returned in i) is one by default, but it is possible to return multiple parameters in one step. For more details, please refer to the "Exploring multiple candidates at once" section of the tutorial. Also, instead of running the above loop inside PHYSBO, it is possible to control i) and ii) separately from the outside. In other words, it is possible to propose the next parameter to be executed from PHYSBO, evaluate its objective function value in some way outside PHYBO (e.g., by experiment rather than numerical calculation), and register the evaluated value in PHYSBO. For more details, please refer to the "Running Interactively" section of the tutorial.  
    
4. Check numerical results

  The search result ``res`` is returned as an object of the ``history`` class ( ``physbo.search.discrete.results.history`` ). The following is a reference to the search results.

  - ``res.fx``: The logs of evaluation values for simulator (objective function) simulator.
  - ``res.chosen_actions``: The logs of the action ID (parameter) when the simulator has executed.
  - ``fbest, best_action= res.export_all_sequence_best_fx()``: The logs of the best values and their action IDs (parameters) at each step where the simulator has executed.
  - ``res.total_num_search``: Total number steps where the simulator has executed.

  The search results can be saved to an external file using the ``save`` method, and the output results can be loaded using the ``load`` method. See the tutorial for details on how to use it.
