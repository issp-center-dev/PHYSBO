# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from matplotlib import pyplot as plt

import physbo
import physbo.gp


seed = 12345
np.random.seed(seed)

N_all = 10 # The number of candidates

X_train = np.random.rand(N_all)
t_train = np.sin(X_train * np.pi)

X_train = X_train.reshape(-1, 1)

X_test = np.linspace(0, 2.0, 101)
t_test = np.sin(X_test * np.pi)
X_test = X_test.reshape(-1, 1)

lik = physbo.gp.lik.gauss()
cov = physbo.gp.cov.gauss(X_train.shape[1], ard = False)
mean = physbo.gp.mean.const()

gp = physbo.gp.model(lik=lik, cov=cov, mean=mean)
config = physbo.misc.set_config()

gp.fit(X_train, t_train, config)
gp.print_params()

gp.prepare(X_train, t_train)
fmean_train = gp.get_post_fmean(X_train, X_train)

MSE_train = np.mean((fmean_train - t_train) ** 2)
print(f"MSE_train: {MSE_train}")

fmean_test = gp.get_post_fmean(X_train, X_test).flatten()
fcov_test = gp.get_post_fcov(X_train, X_test).flatten()

plt.plot(X_test, t_test, "k--", label="true")
plt.scatter(X_train, fmean_train, label="train")
plt.plot(X_test, fmean_test, label="test")
plt.fill_between(
    X_test.flatten(),
    fmean_test - np.sqrt(fcov_test),
    fmean_test + np.sqrt(fcov_test),
    alpha=0.3,
)
plt.legend()
plt.show()
