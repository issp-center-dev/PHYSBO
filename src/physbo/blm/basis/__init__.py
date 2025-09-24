# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._fourier import Fourier

from ...misc import deprecated_warning


def fourier(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.basis.fourier", new="physbo.blm.basis.Fourier")
    return Fourier(*args, **kwargs)

__all__ = ["Fourier"]
