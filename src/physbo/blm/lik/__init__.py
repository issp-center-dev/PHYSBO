# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._cov import Cov
from ._gauss import Gauss
from ._linear import Linear

from ...misc import deprecated_warning


def cov(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.lik.cov", new="physbo.blm.lik.Cov")
    return Cov(*args, **kwargs)


def gauss(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.lik.gauss", new="physbo.blm.lik.Gauss")
    return Gauss(*args, **kwargs)


def linear(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.lik.linear", new="physbo.blm.lik.Linear")
    return Linear(*args, **kwargs)


__all__ = ["Cov", "Gauss", "Linear"]
