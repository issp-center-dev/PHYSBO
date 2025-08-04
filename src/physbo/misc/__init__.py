# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._centering import centering
from ._gauss_elim import gauss_elim
from ._set_config import SetConfig
from ._use_cython import use_cython

from ._src import cholupdate, diagAB, logsumexp64, traceAB2, traceAB3

# from ._src.diagAB import diagAB_64 as diagAB
# from ._src.traceAB import traceAB2_64 as traceAB2
# from ._src.traceAB import traceAB3_64 as traceAB3
# from ._src.cholupdate import cholupdate64 as cholupdate
# from ._src.logsumexp import logsumexp64


__warned_names = set()

def deprecated_warning(old: str, new: str):
    """
    Print a warning message when a deprecated name is used.

    Parameters
    ----------
    old: str
        The old name.
    new: str
        The new name.
    """

    if old not in __warned_names:
        print("-" * 80)
        print(f"WARNING: {old} is deprecated and will be removed in the future.")
        print(f"         Use {new} instead.")
        print("-" * 80)
        __warned_names.add(old)


def set_config(*args,**kwargs):
    deprecated_warning(old="physbo.set_config", new="physbo.SetConfig")
    return SetConfig(*args,**kwargs)
