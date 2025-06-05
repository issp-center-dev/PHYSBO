# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._gauss import Gauss

from ...misc import deprecated_warning


def gauss(*args,**kwargs):
    deprecated_warning(old="physbo.gp.cov.gauss", new="physbo.gp.cov.Gauss")
    return Gauss(*args,**kwargs)
