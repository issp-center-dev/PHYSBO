# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .. import use_cython

if use_cython():
    from physbo_core_cython.misc._src import (
        cholupdate64 as cholupdate,
        diagAB_64 as diagAB,
        logsumexp64 as logsumexp64,
        traceAB2_64 as traceAB2,
        traceAB3_64 as traceAB3,
    )
else:
    from .pure import (
        cholupdate64 as cholupdate,
        diagAB_64 as diagAB,
        logsumexp64 as logsumexp64,
        traceAB2_64 as traceAB2,
        traceAB3_64 as traceAB3,
    )

__all__ = ["cholupdate", "diagAB", "logsumexp64", "traceAB2", "traceAB3"]
