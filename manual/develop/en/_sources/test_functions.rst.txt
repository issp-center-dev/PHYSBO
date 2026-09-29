.. _chap_test_functions:

Benchmark functions
=====================

PHYSBO bundles benchmark (test) functions in :mod:`physbo.test_functions`
for trying out and comparing the optimization algorithms.
The tables below summarize the properties needed to choose a function;
the definition of each function, its references, and the reference box for
the hypervolume are documented in the API reference
(:mod:`physbo.test_functions.multi_objective` and
:mod:`physbo.test_functions.single_objective`), which the class names link to.

Usage
-----

A test function object is callable on an array of points of shape ``(n, dim)``
and returns an array of shape ``(n, nobj)``.
It also provides the search space (``min_X``, ``max_X``), a constraint filter
(``constraint``), a grid generator that applies the constraints (``make_grid``),
and, for multi-objective functions, the reference box for the hypervolume
(``reference_min``, ``reference_max``).

.. code-block:: python

   import physbo

   fn = physbo.test_functions.multi_objective.SRN()
   X = fn.make_grid(101)      # candidates satisfying the constraints
   Y = fn(X)                  # shape (N, 2)
   # ... after the search ...
   vid = res.pareto.volume_in_dominance(fn.reference_min, fn.reference_max)

Each function is implemented in the same sense (minimization or maximization)
as in the reference it follows, shown in the column "Original sense".
Since PHYSBO maximizes objectives, the values returned by a test function
describe a maximization problem by default (``test_maximizer=True``):
the values of a minimization problem are negated, and those of a maximization
problem are returned as they are.
Pass ``test_maximizer=False`` to obtain the values of a minimization problem instead.
The reference box follows the same convention.

Multi-objective functions
-------------------------

The search space of an alias is that of the class it refers to unless stated
otherwise in the column "Aliases".
The Pareto-optimal sets are given for the functions where a closed form is known.

.. include:: _generated/test_functions_multi.rst

Single-objective functions
--------------------------

The global minimum points and values are those of the minimization problem
(``test_maximizer=False``); for the functions with a variable number of
variables they are shown for the default number.

.. include:: _generated/test_functions_single.rst
