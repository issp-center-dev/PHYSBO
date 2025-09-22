# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ....misc import use_cython

if use_cython():
    from physbo_core_cython.gp.cov._src.enhance_gauss import grad_width64, grad_width32
else:
    from .pure import grad_width64, grad_width32

__all__ = ["grad_width64", "grad_width32"]
