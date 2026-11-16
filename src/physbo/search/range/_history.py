# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .._history import _RangeHistory


class History(_RangeHistory):
    """The history of a search for a single objective in a continuous search
    space (physbo.search.range.Policy).

    The observations are in ``fx`` (the objective values) and ``action_X``
    (the inputs); the first ``total_num_search`` rows of these arrays are
    used.

    Parameters
    ----------
    dim: int
        The dimension of the search space.
    """
