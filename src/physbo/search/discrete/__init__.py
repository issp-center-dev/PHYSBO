# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._policy import Policy as Policy
from ._history import History as History

from ...misc import deprecated_warning


def policy(*args, **kwargs):
    deprecated_warning(
        old="physbo.search.discrete.policy", new="physbo.search.discrete.Policy"
    )
    return Policy(*args, **kwargs)


def history(*args, **kwargs):
    deprecated_warning(
        old="physbo.search.discrete.history", new="physbo.search.discrete.History"
    )
    return History(*args, **kwargs)
