# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .._history import _RangeMultiHistory


class History(_RangeMultiHistory):
    """The history of a multi-objective search in a continuous search space
    (physbo.search.range_multi.Policy).

    The observations are in ``fx`` (the objective values, one column per
    objective) and ``action_X`` (the inputs), with one row per observation
    (``total_num_search`` of them). ``pareto`` is the Pareto front of the
    observations.

    Parameters
    ----------
    num_objectives: int
        The number of objectives.
    dim: int
        The dimension of the search space.
    """
