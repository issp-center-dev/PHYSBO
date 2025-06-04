# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._cov import Cov

from ....misc import deprecated_warning

def cov(*args,**kwargs):
    deprecated_warning(old="physbo.blm.lik._src.cov", new="physbo.blm.lik._src.Cov")
    return Cov(*args,**kwargs)
