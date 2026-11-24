# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
import physbo
from pathlib import Path

def load_data():
    # Load data from CSV file
    A = np.asarray(np.loadtxt("data/s5-210.csv", skiprows=1, delimiter=","))
    X = A[:, 0:3]
    t = -A[:, 3]
    return X, t

X, t = load_data()
X = physbo.misc.centering(X)

np.random.seed(12345)
N = len(t)
Ntrain = int(N*0.1)
Ntest = min(int(N*0.1), N-Ntrain)

id_all   = np.random.choice(N, N, replace=False)
id_train  = id_all[0:Ntrain]
id_test = id_all[Ntrain:Ntrain+Ntest]

X_train = X[id_train]
X_test = X[id_test]

t_train = t[id_train]
t_test = t[id_test]

print("Ntrain =", Ntrain)
print("Ntest =", Ntest)

cov = physbo.gp.cov.Gauss( X_train.shape[1],ard = False )
mean = physbo.gp.mean.Const()
lik = physbo.gp.lik.Gauss()
gp = physbo.gp.Model(lik=lik,mean=mean,cov=cov)
config = physbo.misc.SetConfig()
gp.fit(X_train, t_train, config)
gp.print_params()
gp.prepare(X_train, t_train)
fmean = gp.get_post_fmean(X_train, X_test)
fcov = gp.get_post_fcov(X_train, X_test)
mse = np.mean((fmean - t_test) ** 2)
print("Test MSE:", mse)
pi_mean, pi_std = gp.get_permutation_importance(X_train, t_train, n_perm=100)

features = list(range(len(pi_mean)))

plt.figure(figsize=(8, 5))
plt.barh(
    features,
    pi_mean,
    xerr=pi_std,
)
plt.gca().invert_yaxis()
plt.yticks(features)
plt.xlabel("Permutation Importance")
plt.ylabel("Features")
plt.tight_layout()
print("save permutation_importance.pdf")
plt.savefig("permutation_importance.pdf")
plt.show()