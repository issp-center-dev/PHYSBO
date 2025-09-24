# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._zero import Zero
from ._const import Const


from ...misc import deprecated_warning


def zero(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.gp.mean.zero", new="physbo.gp.mean.Zero")
    return Zero(*args, **kwargs)


def const(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.gp.mean.const", new="physbo.gp.mean.Const")
    return Const(*args, **kwargs)

__all__ = ["Zero", "Const"]
