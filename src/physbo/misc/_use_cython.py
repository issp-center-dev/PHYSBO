# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from importlib.util import find_spec

if find_spec("physbo_core_cython"):
    CYTHON_AVAILABLE = True
else:
    CYTHON_AVAILABLE = False

PHYSBO_USE_CYTHON = os.environ.get("PHYSBO_USE_CYTHON", "auto")
if PHYSBO_USE_CYTHON == "0":
    USING_CYTHON = False
elif PHYSBO_USE_CYTHON == "1":
    if not CYTHON_AVAILABLE:
        raise ImportError(
            "PHYSBO_USE_CYTHON is set to 1, but physbo_core_cython is not installed"
        )
    USING_CYTHON = True
else:  # PHYSBO_USE_CYTHON == "auto"
    USING_CYTHON = CYTHON_AVAILABLE

if USING_CYTHON:
    print("Cythonized version of physbo is used")


def use_cython():
    return USING_CYTHON
