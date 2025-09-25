# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from . import gp
from . import opt
from . import blm
from . import misc
from . import search
from . import predictor
from ._variable import Variable


def base_predictor(*args, **kwargs):
    ":meta private:"
    misc.deprecated_warning(old="physbo.base_predictor", new="physbo.predictor.BasePredictor")
    return predictor.BasePredictor(*args, **kwargs)


def variable(*args, **kwargs):
    ":meta private:"
    misc.deprecated_warning(old="physbo.variable", new="physbo.Variable")
    return Variable(*args, **kwargs)


__version__ = "3.0.0"

# __all__ is used to specify the public API of the package
# and is used by sphinx to generate the API documentation
__all__ = ["gp", "opt", "blm", "misc", "search", "predictor", "Variable"]
