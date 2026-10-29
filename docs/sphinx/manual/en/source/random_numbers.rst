.. _chap_random_numbers:

Random numbers and reproducibility
==================================

Several steps of a search draw random numbers: the choice of candidates in
``random_search``, the initial values and the mini-batches of the
hyperparameter learning, the random Fourier basis of the Bayesian linear
model (``num_rand_basis > 0``), the posterior samples of Thompson sampling
(``score="TS"``), the virtual observations used when several points are
proposed per step (``num_search_each_probe > 1``), the random optimizer of
the acquisition function in the range policies, and the weights of
``ParEGO``.

All of these draws go through a random number generator that the policy
selects with the ``rng`` argument of its constructor. The argument is
accepted by every policy (``discrete``, ``discrete_multi``,
``discrete_unified``, ``range``, ``range_multi`` and ``range_unified``).

Two modes
----------------------------------

**Legacy mode (default).**
Without ``rng`` (or with ``rng="legacy"``), the policy draws from the
global ``numpy.random`` state, as all previous versions of PHYSBO did.
``set_seed()`` seeds that global state, so a script that calls
``numpy.random.seed`` or ``policy.set_seed`` keeps giving the same
results as before. The state is shared with everything else in the
process that uses ``numpy.random``.

**Generator mode.**
With ``rng`` given as an integer seed, a ``numpy.random.SeedSequence``, or
a ``numpy.random.Generator``, the policy owns a ``numpy.random.Generator``
and draws only from it. The global ``numpy.random`` state is neither read
nor modified, so the search is reproducible regardless of what else the
process does with ``numpy.random``, and the generator is part of the
policy object: pickling the policy stores its state (see
:ref:`chap_checkpoint`). In this mode ``set_seed(seed)`` re-creates the generator from
the seed. The random number stream differs from the legacy one, so a run
in the Generator mode does not reproduce a run of a previous version
with the same seed.

.. code-block:: python

    # legacy mode: the global numpy.random state, seeded by set_seed()
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(0)

    # Generator mode: a policy-owned generator seeded with 12345
    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)

    # Generator mode with a generator made by the caller
    rng = numpy.random.default_rng(12345)
    policy = physbo.search.discrete.Policy(test_X=X, rng=rng)

In either mode, the same seed gives the same search on the same version
of PHYSBO in the same environment (Python, NumPy and SciPy versions,
BLAS/LAPACK library).

Helper objects
----------------------------------

Objects passed to the policy that draw random numbers themselves follow
the policy's generator unless they were given one of their own:

- ``physbo.search.unify.ParEGO`` draws its weights through its ``rng``
  attribute. When it is given to ``bayes_search`` as ``unify_method``
  without an ``rng`` argument, it uses the generator of the policy; with
  ``ParEGO(..., rng=...)`` it keeps the given generator.
- The default random optimizer of the range policies is created with the
  generator of the policy. An optimizer created by the caller takes its
  own ``rng`` argument, e.g.
  ``physbo.search.optimize.random.Optimizer(min_X, max_X, nsamples=1000, rng=policy.rng)``.

Parallel runs
----------------------------------

In a parallel run (a communicator passed as ``comm``), give the same
``rng`` specification to the policy on every rank. The quantities that
all ranks must agree on are drawn on rank 0 and broadcast: the
hyperparameters of the model, the random Fourier basis, the posterior
sample of Thompson sampling with the Bayesian linear model, the virtual
observations of a multi-point proposal, the point returned by the random
optimizer, and the result of the simulator. The remaining draws are made
by each rank for its own share of the candidates. As a consequence, the
random number streams of the ranks do not stay in step with each other
(rank 0 draws more than the others); this is harmless, because every
quantity that affects the search on more than one rank is broadcast.

With the Bayesian linear model (``num_rand_basis > 0``), Thompson
sampling therefore proposes the same points as a serial run with the
same seed, whatever the number of ranks. With the Gaussian process
predictor (``num_rand_basis == 0``) the posterior sample has no finite
representation that could be broadcast, so each rank samples the
posterior of its own candidates only: the correlations between
candidates on different ranks are ignored, and the proposals depend on
the number of ranks. For Thompson sampling in parallel runs the Bayesian
linear model is recommended.

``get_score`` with ``parallel=True`` (the default) splits the candidates
over the ranks and draws the Thompson sample on rank 0; with
``parallel=False`` each rank scores the candidates it was given with a
sample of its own.
