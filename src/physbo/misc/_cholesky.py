# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

def cholupdate(L, x):
    """Cholesky update

    L is updated in place.
    x is not updated.
    """

    N = x.shape[0]
    x2 = x.copy()

    for k in range(N):
        r = np.hypot(L[k, k], x2[k])
        c = r / L[k, k]
        s = x2[k] / L[k, k]
        L[k, k] = r

        L[k, k+1:] += s * x2[k+1:]
        L[k, k+1:] /= c
        x2[k+1:] *= c
        x2[k+1:] -= s * L[k, k+1:]
