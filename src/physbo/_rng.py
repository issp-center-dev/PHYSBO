# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


class LegacyRNG:
    """Random number adapter backed by the global ``numpy.random`` state.

    This class exposes the method names of ``numpy.random.Generator`` so
    that the internal code can be written against the Generator API while
    remaining bit-compatible with the historical behavior of PHYSBO, which
    draws from the global ``numpy.random`` state (seeded via
    ``numpy.random.seed``).
    """

    def seed(self, seed=None):
        np.random.seed(seed)

    def standard_normal(self, size=None):
        return np.random.standard_normal(size)

    def random(self, size=None):
        return np.random.random_sample(size)

    def integers(self, low, high=None, size=None):
        return np.random.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return np.random.choice(a, size, replace, p)

    def permutation(self, x):
        return np.random.permutation(x)

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.random.uniform(low, high, size)

    def multivariate_normal(self, mean, cov, size=None):
        return np.random.multivariate_normal(mean, cov, size)


_default_rng = LegacyRNG()


def get_rng(rng=None):
    """Normalize an rng argument.

    Parameters
    ----------
    rng: None or rng object
        If None (default), the module-wide legacy RNG backed by the global
        ``numpy.random`` state is returned. Otherwise ``rng`` is returned
        as is (e.g. a ``numpy.random.Generator`` or a ``LegacyRNG``).

    Returns
    -------
    rng object
    """
    if rng is None:
        return _default_rng
    return rng


def make_rng(rng=None):
    """Normalize the user-facing ``rng`` argument of the policies.

    Parameters
    ----------
    rng: None, "legacy", int, numpy.random.SeedSequence, or rng object
        - None or "legacy" (default): the legacy RNG backed by the global
          ``numpy.random`` state. Fully bit-compatible with the historical
          behavior of PHYSBO (``set_seed()`` seeds the global state).
        - int or ``numpy.random.SeedSequence``: a new
          ``numpy.random.Generator`` seeded with it. The random number
          stream differs from the legacy one, and the state is stored on
          the policy (included in pickle-based checkpoints).
        - rng object: used as is (e.g. a ``numpy.random.Generator``).

    Returns
    -------
    rng object
    """
    if rng is None:
        return _default_rng
    if isinstance(rng, str):
        if rng == "legacy":
            return _default_rng
        raise ValueError(
            f"Unknown rng specification: {rng!r} (only 'legacy' is allowed as str)"
        )
    if isinstance(rng, (int, np.integer, np.random.SeedSequence)):
        return np.random.default_rng(rng)
    return rng


def attach_rng(obj, rng):
    """Let a helper object draw from the policy's random number generator.

    Objects that draw random numbers on behalf of a policy (for example
    ``physbo.search.unify.ParEGO``) hold their generator in an ``rng``
    attribute. When that attribute is the legacy default, it is replaced
    by ``rng`` so that the object follows the mode of the policy: in the
    Generator mode its draws then come from the policy-owned generator
    and are covered by the policy's pickled state. An object without an
    ``rng`` attribute, or one that was given a generator of its own, is
    left unchanged.

    Parameters
    ----------
    obj: object
        The helper object (e.g. a unify_method).
    rng: rng object
        The policy's random number generator.
    """
    if isinstance(getattr(obj, "rng", None), LegacyRNG):
        obj.rng = rng
