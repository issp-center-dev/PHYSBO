# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .._history import _DiscreteMultiHistory


class History(_DiscreteMultiHistory):
    """The history of a search with unified objectives in a discrete search space
    (physbo.search.discrete_unified.Policy).

    It records the objectives themselves, not the unified one. The observations
    are in ``fx`` (the objective values, one column per objective) and
    ``chosen_actions`` (the indexes of the actions), with one row per
    observation (``total_num_search`` of them). ``pareto`` is the Pareto front
    of the observations.

    Parameters
    ----------
    num_objectives: int
        The number of objectives.
    """
