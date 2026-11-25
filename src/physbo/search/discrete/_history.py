# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .._history import _DiscreteHistory


class History(_DiscreteHistory):
    """The history of a search for a single objective in a discrete search
    space (physbo.search.discrete.Policy).

    The observations are in ``fx`` (the objective values) and
    ``chosen_actions`` (the indexes of the actions); the first
    ``total_num_search`` entries of these arrays are used.
    """
