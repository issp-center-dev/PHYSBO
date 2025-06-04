# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._src import Cov
from ._linear import Linear
from ._gauss import Gauss

from ...misc import deprecated_warning

def linear(*args,**kwargs):
    deprecated_warning(old="physbo.blm.lik.linear", new="physbo.blm.lik.Linear")
    return Linear(*args,**kwargs)


def gauss(*args,**kwargs):
    deprecated_warning(old="physbo.blm.lik.gauss", new="physbo.blm.lik.Gauss")
    return Gauss(*args,**kwargs)

def cov(*args,**kwargs):
    deprecated_warning(old="physbo.blm.lik.cov", new="physbo.blm.lik.Cov")
    return Cov(*args,**kwargs)
