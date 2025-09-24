# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Bayesian Linear Model"""

from . import basis
from . import prior
from . import lik
from . import inf
from . import core

from .core import Model
from ._predictor import Predictor

from ..misc import deprecated_warning


def model(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.model", new="physbo.blm.Model")
    return Model(*args, **kwargs)


def predictor(*args, **kwargs):
    ":meta private:"
    deprecated_warning(old="physbo.blm.predictor", new="physbo.blm.Predictor")
    return Predictor(*args, **kwargs)


__all__ = ["basis", "prior", "lik", "inf", "core", "Model", "Predictor"]
