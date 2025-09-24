# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._policy import Policy
from ._history import History

from ...misc import deprecated_warning


def policy(*args, **kwargs):
    ":meta private:"
    deprecated_warning(
        old="physbo.search.discrete_multi.policy",
        new="physbo.search.discrete_multi.Policy",
    )
    return Policy(*args, **kwargs)


def history(*args, **kwargs):
    ":meta private:"
    deprecated_warning(
        old="physbo.search.discrete_multi.history",
        new="physbo.search.discrete_multi.History",
    )
    return History(*args, **kwargs)

__all__ = ["Policy", "History"]
