# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._gauss import Gauss as Gauss

from ...misc import deprecated_warning


def gauss(*args, **kwargs):
    deprecated_warning(old="physbo.blm.prior.gauss", new="physbo.blm.prior.Gauss")
    return Gauss(*args, **kwargs)
