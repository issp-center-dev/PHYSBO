# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Smoke tests running the benchmark driver in tests/benchmark with small
parameters.

Besides guarding the benchmark scripts themselves, these tests exercise
code paths that the other tests do not reach: make_grid, the hypervolume
calculation on real search results, the ParEGO/NDS unifying methods, and
the random optimizer of the range policies.
"""

import os
import sys

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))
import common  # noqa: E402


@pytest.mark.parametrize("score", ["EI", "PI", "TS"])
@pytest.mark.parametrize("unify_method_name", ["None", "ParEGO", "NDS"])
@pytest.mark.parametrize("use_range", [False, True])
def test_benchmark_smoke(score, unify_method_name, use_range):
    fn = physbo.test_functions.multi_objective.VLMOP2()
    params = common.set_parameters(
        use_range=use_range,
        fn=fn,
        N=11,
        score=score,
        nrand_basis_TS=50,
        unify_method_name=unify_method_name,
    )
    vid, elapsed_time, num_bayes = common.benchmark(
        fn=fn,
        score=params["score"],
        num_random_search=5,
        num_bayes_search=2,
        num_bayes_search_set=1,
        nrand_basis=params["nrand_basis"],
        N=11,
        seed=12345,
        output_dir="",
        pdffilename_prefix=None,
        optimizer=params["optimizer"],
        unify_method=params["unify_method"],
    )
    assert len(vid) == 1
    assert len(elapsed_time) == 1
    assert num_bayes == [2]
    # the searched points must dominate a positive volume
    assert vid[0] > 0.0
    assert vid[0] <= np.prod(np.array(fn.reference_max) - np.array(fn.reference_min))


def test_benchmark_gen_centers():
    c1 = common.gen_centers(nobj=3, dim=1)
    assert c1.shape == (3, 1)

    c2 = common.gen_centers(nobj=4, dim=2)
    assert c2.shape == (4, 2)
    # centers are rotations of each other: all on the unit circle
    np.testing.assert_allclose(np.linalg.norm(c2, axis=1), np.ones(4))

    with pytest.raises(ValueError):
        common.gen_centers(nobj=2, dim=3)


def test_benchmark_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    fn = physbo.test_functions.multi_objective.VLMOP2()
    params = common.set_parameters(
        use_range=False,
        fn=fn,
        N=11,
        score="EI",
        nrand_basis_TS=50,
        unify_method_name="None",
    )
    common.benchmark(
        fn=fn,
        score=params["score"],
        num_random_search=5,
        num_bayes_search=2,
        num_bayes_search_set=1,
        nrand_basis=params["nrand_basis"],
        N=11,
        seed=12345,
        output_dir=str(tmp_path),
        pdffilename_prefix="solutions_smoke",
        optimizer=params["optimizer"],
        unify_method=params["unify_method"],
    )
    assert (tmp_path / "solutions_smoke_2.pdf").exists()
