# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._adam import Adam

from ..misc import deprecated_warning


def adam(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.opt.adam", new="physbo.opt.Adam")
    return Adam(*args, **kwargs)

__all__ = ["Adam"]
