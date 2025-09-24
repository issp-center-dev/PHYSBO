# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from . import gp as gp
from . import opt as opt
from . import blm as blm
from . import misc as misc
from . import search as search
from . import predictor as predictor
from .predictor import BasePredictor as BasePredictor
from ._variable import Variable as Variable


def base_predictor(*args, **kwargs):
    misc.deprecated_warning(old="physbo.base_predictor", new="physbo.BasePredictor")
    return BasePredictor(*args, **kwargs)


def variable(*args, **kwargs):
    misc.deprecated_warning(old="physbo.variable", new="physbo.Variable")
    return Variable(*args, **kwargs)


__version__ = "3.0-dev"
