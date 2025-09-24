# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from . import cov as cov
from . import mean as mean
from . import lik as lik
from . import core as core

from .core import Prior as Prior
from .core import Model as Model
from .core import SFS as SFS
from .core import learning as learning
from ._predictor import Predictor as Predictor

from ..misc import deprecated_warning


def prior(*args, **kwargs):
    deprecated_warning(old="physbo.gp.prior", new="physbo.gp.Prior")
    return Prior(*args, **kwargs)


def model(*args, **kwargs):
    deprecated_warning(old="physbo.gp.model", new="physbo.gp.Model")
    return Model(*args, **kwargs)


def sfs(*args, **kwargs):
    deprecated_warning(old="physbo.gp.sfs", new="physbo.gp.SFS")
    return SFS(*args, **kwargs)


def predictor(*args, **kwargs):
    deprecated_warning(old="physbo.gp.predictor", new="physbo.gp.Predictor")
    return Predictor(*args, **kwargs)
