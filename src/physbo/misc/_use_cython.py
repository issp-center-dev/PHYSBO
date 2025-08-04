# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from .._build_info import USING_CYTHON


def use_cython():
    use_cython = os.environ.get("PHYSBO_USE_CYTHON", "auto")
    if use_cython == "0":
        return False
    elif use_cython == "1":
        return True
    else: # use_cython == "auto"
        return USING_CYTHON
