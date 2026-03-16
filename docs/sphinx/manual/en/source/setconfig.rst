Detailed Settings for Learning
===============================

PHYSBO performs hyperparameter learning ("learning") for the Gaussian process, but the learning method itself has hyperparameters such as the learning rate for Adam.
Use ``physbo.SetConfig`` to change these hyperparameters.
In most cases, you do not need to change the default values.

Usage of ``SetConfig``
----------------------

First, create an instance with ``config = physbo.SetConfig()``. All parameters have default values, so modify only the parameters you want to change.
Pass the created ``config`` to the ``config`` argument of the ``Policy`` constructor.
For example, to set Adam's learning rate to 0.01:

.. code-block:: python

   config = physbo.SetConfig()
   config.adam.alpha = 0.01
   policy = physbo.search.discrete.Policy(test_X, config=config)

You can also save parameters to an INI file and load them with ``config.load("config.ini")``.
For example, create an INI file like the following:

.. code-block:: ini

   [learning.adam]
   alpha = 0.01


Configurable Parameters
-----------------------

The INI file has a hierarchical structure of sections and keys.
Accordingly, ``SetConfig`` also has a hierarchical structure.
For example, the ``alpha`` key in the ``[learning.adam]`` section corresponds to the ``config.learning.alpha`` member variable.

Details of sections and parameters are as follows.

``[learning]`` -- Common learning settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common settings for Gaussian process regression model learning.
Currently, the learning method ``method`` supports three types: adam, bfgs, and batch.
Adam is for online learning; bfgs and batch are for batch learning.

.. csv-table::
   :header: Key, Type, Default, Description
   :widths: 26, 8, 12, 60

   ``method``, str, "adam", "Learning method. One of ``adam``, ``bfgs``, ``batch``"
   ``is_disp``, bool, True, "Whether to display during learning. One of ``true``, ``false``"
   ``num_disp``, int, 10, Display interval (how many iterations between displays)
   ``num_init_params_search``, int, 20, Number of iterations for initial hyperparameter search

``[learning.online]`` -- Common online learning settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common settings for online learning.
Used when ``method=adam``.

.. csv-table::
   :header: Key, Type, Default, Description
   :widths: 32, 8, 10, 42

   ``max_epoch``, int, 500, Maximum number of epochs
   ``max_epoch_init_params_search``, int, 50, Maximum number of epochs for initial hyperparameter search
   ``batch_size``, int, 64, Batch size
   ``eval_size``, int, 5000, Number of samples for evaluation

``[learning.adam]`` -- Adam learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adam settings.
Used when ``method=adam``.

.. csv-table::
   :header: Key, Type, Default, Description
   :widths: 12, 8, 10, 42

   ``alpha``, float, 0.001, Learning rate
   ``beta``, float, 0.9, Decay rate for first moment
   ``gamma``, float, 0.999, Decay rate for second moment
   ``epsilon``, float, 1e-6, Small constant for numerical stability

``[learning.batch]`` -- Batch / BFGS learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Settings for bfgs and batch.
Used when ``method=bfgs`` or ``method=batch``.

.. csv-table::
   :header: Key, Type, Default, Description
   :widths: 32, 8, 10, 42

   ``max_iter``, int, 200, Maximum number of iterations
   ``max_iter_init_params_search``, int, 20, Maximum number of iterations for initial hyperparameter search
   ``batch_size``, int, 5000, Batch size


``[search]`` -- Parameters used in Bayesian optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: Key, Type, Default, Description
   :widths: 28, 8, 10, 60

   ``multi_probe_num_sampling``, int, 20, Used when ``num_search_each_probe>1``. Number of samples for computing unevaluated objective function values
   ``alpha``, float, 1.0, Thompson Sampling hyperparameter. Coefficient to rescale the standard deviation of the posterior distribution during sampling
