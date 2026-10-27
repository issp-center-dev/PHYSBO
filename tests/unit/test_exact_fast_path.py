# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The fast path of the exact inference for the default model agrees with the
generic path."""

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")

from physbo.gp.inf import exact


def make_model(ard, mean_name, num_dim):
    mean = physbo.gp.mean.Const() if mean_name == "const" else physbo.gp.mean.Zero()
    return physbo.gp.core.Model(
        lik=physbo.gp.lik.Gauss(),
        mean=mean,
        cov=physbo.gp.cov.Gauss(num_dim, ard=ard),
    )


def generic(fn, monkeypatch):
    """Evaluate fn with the fast path disabled."""
    with monkeypatch.context() as m:
        m.setattr(exact, "_uses_gauss_fast_path", lambda gp, params=None: False)
        return fn()


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
@pytest.mark.parametrize("N", [1, 7, 60])
def test_fast_path_matches_generic(ard, mean_name, N, monkeypatch):
    rng = np.random.default_rng(N + 10 * ard)
    num_dim = 3
    X = rng.random((N, num_dim)) * 4.0 + 100.0  # offset: distances stay accurate
    t = np.sin(X[:, 0]) + 0.1 * rng.normal(size=N)
    gp = make_model(ard, mean_name, num_dim)
    assert exact._uses_gauss_fast_path(gp)
    np.random.seed(N)  # get_cand_params samples pairs of points
    base = gp.get_cand_params(X, t) if N > 1 else gp.params.copy()
    for trial in range(5):
        params = base + rng.normal(scale=0.5, size=base.shape)
        if trial == 4:
            # a kernel width beyond its bound: clipped in place by both paths
            params[gp.lik.num_params + gp.prior.mean.num_params] = 20.0
        p_fast = params.copy()
        p_gen = params.copy()
        f_fast = gp.eval_marlik(p_fast, X, t)
        g_fast = gp.get_grad_marlik(p_fast, X, t)
        f_gen = generic(lambda: gp.eval_marlik(p_gen, X, t), monkeypatch)
        g_gen = generic(lambda: gp.get_grad_marlik(p_gen, X, t), monkeypatch)
        # the condition number of A reaches ~1e5 here
        assert f_fast == pytest.approx(f_gen, rel=1e-9, abs=1e-9)
        numpy.testing.assert_allclose(
            g_fast, g_gen, rtol=1e-9, atol=1e-9 * max(1.0, np.max(np.abs(g_gen)))
        )
        # the same side effects on the parameter array
        numpy.testing.assert_array_equal(p_fast, p_gen)


def test_fast_path_not_used_for_subclasses():
    class MyGauss(physbo.gp.cov.Gauss):
        pass

    gp = physbo.gp.core.Model(
        lik=physbo.gp.lik.Gauss(), mean=physbo.gp.mean.Const(), cov=MyGauss(2)
    )
    assert not exact._uses_gauss_fast_path(gp)


@pytest.mark.parametrize(
    "num_dim, ard, X",
    [
        # ARD with fewer or more columns than widths: the generic path
        # broadcasts X against the widths (or rejects it)
        (2, True, [[0.0], [1.0], [0.3]]),
        (1, True, [[0.0, 1.0], [1.0, 0.5], [0.3, 0.2]]),
        (2, True, [[0.0, 1.0, 2.0], [1.0, 0.5, 0.1]]),
        # X not two-dimensional
        (1, False, [0.0, 1.0, 0.3]),
        (1, True, [0.0, 1.0, 0.3]),
    ],
)
def test_fast_path_input_shapes_follow_generic(num_dim, ard, X, monkeypatch):
    gp = make_model(ard, "const", num_dim)
    X = np.array(X)
    t = np.array([0.0, 0.4, -0.2])[: X.shape[0]]
    params = np.zeros(gp.num_params)
    assert not exact._inputs_allow_fast_path(gp, X, t)
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    for fast, gen in ((f_fast, f_gen), (g_fast, g_gen)):
        if isinstance(gen, type):
            assert fast is gen
        else:
            numpy.testing.assert_array_equal(fast, gen)


LONGDOUBLE_IS_DOUBLE = pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8, reason="longdouble is float64 on this platform"
)


@pytest.mark.parametrize(
    "make_X, make_t",
    [
        # targets of another dtype, shape or array class, and np.matrix X:
        # the generic path rejects them (on some NumPy/SciPy versions)
        (np.asarray, lambda t: t.astype(object)),
        (np.asarray, lambda t: np.ma.masked_array(t, mask=[0, 1, 0, 0])),
        (np.asarray, lambda t: t[None, :]),
        (np.matrix, np.asarray),
        # extended precision: SciPy warns about it in the generic path
        pytest.param(np.asarray, lambda t: t.astype(np.longdouble), marks=LONGDOUBLE_IS_DOUBLE),
        pytest.param(lambda X: X.astype(np.longdouble), np.asarray, marks=LONGDOUBLE_IS_DOUBLE),
    ],
)
def test_fast_path_input_types_follow_generic(make_X, make_t, monkeypatch):
    gp = make_model(True, "const", 2)
    X = make_X(np.array([[0.0, 1.0], [0.2, 0.4], [0.9, 0.1], [0.5, 0.5]]))
    t = make_t(np.array([0.3, -0.1, 0.4, 0.0]))
    params = np.array([np.log(0.1), 0.0, 0.0, 0.0, 0.0])
    assert not exact._inputs_allow_fast_path(gp, X, t)
    for method in ("eval_marlik", "get_grad_marlik"):
        fast = run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t))
        gen = generic(
            lambda: run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t)),
            monkeypatch,
        )
        assert fast[1] == gen[1]
        if isinstance(gen[0], type):
            assert fast[0] is gen[0]
        else:
            numpy.testing.assert_array_equal(fast[0], gen[0])


@pytest.mark.parametrize(
    "make_X",
    [
        lambda X: X.astype(np.float32),
        lambda X: X.astype(np.int64),
        np.asfortranarray,
        lambda X: np.repeat(X, 2, axis=0)[::2],
    ],
)
def test_fast_path_taken_for_common_arrays(make_X):
    gp = make_model(True, "const", 2)
    X = make_X(np.array([[0.0, 1.0], [2.0, 4.0], [9.0, 1.0]]))
    for t in (np.array([0.3, -0.1, 0.4]), np.array([1, 0, 2]), np.zeros(3, np.float32)):
        assert exact._inputs_allow_fast_path(gp, X, t)


@pytest.mark.parametrize("log_std", [-400.0, 400.0, np.inf, np.nan])
def test_selecting_the_path_does_not_warn_or_raise(log_std):
    # the noise variance of the gate underflows or overflows; the selected
    # path then warns or raises as the generic path does
    import warnings

    gp = make_model(False, "zero", 1)
    params = np.array([log_std, 0.0, 0.0])
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        exact._uses_gauss_fast_path(gp, params)


def test_fast_path_float16_inputs_ard(monkeypatch):
    # the differences of float16 coordinates are promoted to float64 when
    # divided by the widths, as in the generic path (also with NumPy 1)
    gp = make_model(True, "const", 2)
    X = np.array([[0.0, 1.0], [0.2, 0.4], [0.9, 0.1], [0.5, 0.5], [0.33, 0.71]], dtype=np.float16)
    t = np.array([0.3, -0.1, 0.4, 0.0, 0.2])
    params = np.array([np.log(0.1), 0.0, np.log(0.5), np.log(2.0), 0.2])
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-10)


def test_fast_path_isotropic_any_number_of_columns(monkeypatch):
    # the isotropic kernel takes the distance over all columns of X in both
    # paths, whatever num_dim is
    gp = make_model(False, "const", 1)
    X = np.array([[0.0, 1.0, 2.0], [1.0, 0.5, 0.1], [0.3, 0.2, 0.9]])
    t = np.array([0.0, 0.4, -0.2])
    params = np.array([np.log(0.1), 0.0, 0.0, 0.0])
    assert exact._inputs_allow_fast_path(gp, X, t)
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-10)


def test_fast_path_complex_targets_follow_generic(monkeypatch):
    gp = make_model(False, "const", 1)
    X = np.array([[0.0], [1.0]])
    t = np.array([0.0, 1.0 + 0.5j])
    params = np.zeros(gp.num_params)
    assert not exact._inputs_allow_fast_path(gp, X, t)
    for method in ("eval_marlik", "get_grad_marlik"):
        fast = run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t))
        gen = generic(
            lambda: run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t)),
            monkeypatch,
        )
        assert fast[1] == gen[1]
        numpy.testing.assert_array_equal(fast[0], gen[0])


def compare_both_paths(gp, X, t, params, monkeypatch):
    """Value, gradient or exception type of both paths."""

    def run(fn):
        try:
            return fn()
        except Exception as e:  # compared below
            return type(e)

    out = []
    for path in ("fast", "generic"):
        p = None if params is None else params.copy()
        if path == "fast":
            out.append(
                (run(lambda: gp.eval_marlik(p, X, t)), run(lambda: gp.get_grad_marlik(p, X, t)))
            )
        else:
            out.append(
                (
                    generic(lambda: run(lambda: gp.eval_marlik(p, X, t)), monkeypatch),
                    generic(lambda: run(lambda: gp.get_grad_marlik(p, X, t)), monkeypatch),
                )
            )
    return out


@pytest.mark.parametrize("ard", [False, True])
def test_fast_path_large_offset(ard, monkeypatch):
    # the differences of the coordinates are taken before scaling by the
    # width, as in the generic path
    gp = make_model(ard, "const", 1)
    X = np.array([[1e16], [1e16 + 2.0], [1e16 + 8.0]])
    t = np.array([0.0, 1.0, 0.5])
    params = np.array([np.log(0.1), 0.2, np.log(3.0), 0.0])
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-10)


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
@pytest.mark.parametrize(
    "lik_cov_params",
    [
        ([-30.0], [0.0, 30.0]),  # tiny noise and the largest scale
        ([-30.0], [0.0, 13.8]),
        ([np.log(1e-6)], [0.0, 0.0]),  # lower bound of the noise
        ([np.log(1e-4)], [-1.0, 1.0]),
    ],
)
def test_fast_path_nearly_singular(ard, mean_name, lik_cov_params, monkeypatch):
    # duplicated and close points: both paths succeed or raise the same
    # exception
    gp = make_model(ard, mean_name, 1)
    lik_params, cov_params = lik_cov_params
    mean_params = [0.1] if mean_name == "const" else []
    params = np.array(lik_params + mean_params + cov_params)
    X = np.array([[0.0], [0.0], [1e-9], [0.5]])
    t = np.array([0.0, 0.1, 0.2, 0.3])
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    # the marginal likelihood: the same factorization and triangular solve
    if isinstance(f_gen, type):
        assert f_fast is f_gen
    else:
        assert f_fast == pytest.approx(f_gen, rel=1e-8)
    # the gradient: when A is extremely ill-conditioned (an accuracy cutoff,
    # cond(A) >= 1e12; the two parameter sets with noise std e^-30 reach
    # 1e18-1e19), whether the factorization or the inversion fails and the
    # values otherwise depend on the LAPACK rounding and differ between SciPy
    # versions, so only the failure or the finiteness of both is compared
    # (test_fast_path_gradient_near_cutoff checks the accuracy below the
    # cutoff).  A is built as in the generic path; the noise parameter is not
    # bounded there, unlike the kernel parameters
    G = gp.prior.get_cov(X, params=params[gp.lik.num_params :].copy())
    eye = np.identity(X.shape[0])
    cond = np.linalg.cond(G + np.exp(2 * params[0]) * eye + 1e-8 * eye)
    if isinstance(g_gen, type):
        assert g_fast is g_gen
    elif not cond < 1e12:
        assert not isinstance(g_fast, type)
        assert np.all(np.isfinite(g_fast)) and np.all(np.isfinite(g_gen))
    else:
        numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-6, atol=1e-6 * np.max(np.abs(g_gen)))


def test_fast_path_default_params(monkeypatch):
    rng = np.random.default_rng(5)
    X = rng.random((12, 2))
    t = rng.normal(size=12)
    gp = make_model(True, "const", 2)
    (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, None, monkeypatch)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-10, atol=1e-12)


def test_fast_path_noise_overflow_follows_numpy(monkeypatch):
    # exp(400) is finite but its square overflows; both paths use NumPy
    gp = make_model(False, "zero", 1)
    X = np.array([[0.0], [1.0]])
    t = np.array([0.0, 1.0])
    params = np.array([400.0, 0.0, 0.0])
    with np.errstate(over="raise"):
        (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    assert f_fast is FloatingPointError and f_gen is FloatingPointError
    assert g_fast is FloatingPointError and g_gen is FloatingPointError


@pytest.mark.parametrize(
    "policy",
    [
        {"over": "raise"},
        {"over": "ignore", "invalid": "raise"},
        {"over": "ignore", "invalid": "ignore"},
    ],
)
@pytest.mark.parametrize("log_std", [354.8, 400.0])  # 2 var or var overflows
def test_fast_path_huge_noise_follows_numpy(log_std, policy, monkeypatch):
    gp = make_model(False, "zero", 1)
    X = np.array([[0.0], [1.0]])
    t = np.array([0.0, 1.0])
    params = np.array([log_std, 0.0, 0.0])
    with np.errstate(**policy):
        (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    for fast, gen in ((f_fast, f_gen), (g_fast, g_gen)):
        if isinstance(gen, type):
            assert fast is gen
        else:
            numpy.testing.assert_array_equal(fast, gen)


# gradients computed with 60-digit arithmetic (mpmath, numerical
# differentiation of the marginal likelihood) from the float64 values of X, t
# and the parameters, for A built from the same kernel, noise variance and
# 1e-8 jitter; the digits beyond the tolerances are not significant
HIGH_PRECISION_CASES = [
    # kernel-dominated: duplicated points and noise std 1e-6; the accuracy of
    # both paths is limited by the condition number of A
    (
        [[0.0], [0.0], [1e-9], [0.5]],
        [0.0, 0.1, 0.2, 0.3],
        [np.log(1e-6), 0.1, 0.0, 0.0],
        "const",
        False,
        [-199.9598052190372, -0.10389815259861027, -0.7210516922451755, 1.8191555157744086],
        1e-3,
    ),
    # noise-dominated: noise std 1e6, scale 1e-6
    (
        [[0.0]],
        [0.0],
        [np.log(1e6), 0.0, np.log(1e-6)],
        "zero",
        False,
        [1.0, 0.0, 1.000000000000002e-24],
        1e-12,
    ),
    (
        [[0.0], [0.3], [1.0]],
        [0.5, -0.2, 0.1],
        [np.log(1e3), 0.0, np.log(1e-3)],
        "zero",
        False,
        [2.99999969999697, -1.405280149444326e-20, 2.999999861847796e-12],
        1e-12,
    ),
    (
        [[0.0, 1.0], [0.2, 0.4], [0.9, 0.1], [0.5, 0.5]],
        [0.3, -0.1, 0.4, 0.0],
        [np.log(0.1), 0.05, np.log(0.5), np.log(2.0), 0.2],
        "const",
        True,
        [0.16641031439349524, -0.4168775874424377, -1.4973809618631588, -0.3520613038474567, 3.02573426978047],
        1e-12,
    ),
    # close points with ARD
    (
        [[0.0, 1.0], [1e-5, 1.0], [0.9, 0.1], [0.9, 0.1 + 1e-6]],
        [0.3, 0.31, 0.4, 0.41],
        [np.log(1e-5), 0.05, np.log(0.5), np.log(2.0), 0.2],
        "const",
        True,
        [-95.48116189729322, 2.3034872723319384, 156.82582495491584, 3.9617611832868307, -123.1227867724739],
        1e-5,
    ),
]


@pytest.mark.parametrize("X, t, params, mean_name, ard, ref, rtol", HIGH_PRECISION_CASES)
def test_fast_path_gradient_against_high_precision(X, t, params, mean_name, ard, ref, rtol):
    X = np.array(X)
    gp = make_model(ard, mean_name, X.shape[1])
    grad = gp.get_grad_marlik(np.array(params), X, np.array(t))
    # component by component; exact zeros are compared absolutely
    for g, r in zip(grad, ref):
        assert g == pytest.approx(r, rel=rtol, abs=1e-30 if r != 0 else 1e-15)


# just below the cutoff of test_fast_path_nearly_singular (references computed
# as above); the error of both paths grows like cond(A) * eps (up to 7e-5 here)
# relative to the largest component, and is about 1e-1 at cond(A) = 1e12
NEAR_CUTOFF_CASES = [
    # duplicated points, scale e^2.5: cond(A) = 5.7e10
    (
        [[0.0], [0.0], [1e-9], [0.5]],
        [0.0, 0.1, 0.2, 0.3],
        [np.log(1e-6), 0.1, 0.0, 2.5],
        "const",
        False,
        [-199.95980486595488, 0.0016278666284814715, -0.875693815592253, 1.9970039703859708],
    ),
    # close points with ARD, scale e^3: cond(A) = 9.5e10
    (
        [[0.0, 1.0], [1e-5, 1.0], [0.9, 0.1], [0.9, 0.1 + 1e-6]],
        [0.3, 0.31, 0.4, 0.41],
        [np.log(1e-6), 0.05, np.log(0.5), np.log(2.0), 3.0],
        "const",
        True,
        [-0.4982778828765635, 0.3384366593483278, 618.4655674438284, 68.61546552954869, -560.4588577075933],
    ),
]


@pytest.mark.parametrize("X, t, params, mean_name, ard, ref", NEAR_CUTOFF_CASES)
def test_fast_path_gradient_near_cutoff(X, t, params, mean_name, ard, ref, monkeypatch):
    X = np.array(X)
    t = np.array(t)
    ref = np.array(ref)
    gp = make_model(ard, mean_name, X.shape[1])
    g_fast = gp.get_grad_marlik(np.array(params), X, t)
    g_gen = generic(lambda: gp.get_grad_marlik(np.array(params), X, t), monkeypatch)
    for grad in (g_fast, g_gen):
        numpy.testing.assert_allclose(grad, ref, rtol=0, atol=3e-4 * np.max(np.abs(ref)))


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
@pytest.mark.parametrize(
    "t0, log_scale",
    [
        (1e147, np.log(1e-6)),  # A ~ 1e-8, alpha ~ 1e155: alpha^T alpha overflows
        (1e154, 0.0),  # the scale gradient overflows in the generic path
    ],
)
def test_fast_path_large_targets(ard, mean_name, t0, log_scale, monkeypatch):
    gp = make_model(ard, mean_name, 1)
    X = np.array([[0.0]])
    t = np.array([t0])
    mean_params = [0.0] if mean_name == "const" else []
    params = np.array([np.log(1e-6)] + mean_params + [0.0, log_scale])
    with np.errstate(over="ignore"):
        (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    assert np.isfinite(f_gen)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    # the same finite values and the same infinities
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-12)


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
@pytest.mark.parametrize("policy", [{"over": "ignore", "invalid": "ignore"}, {"all": "raise"}])
def test_fast_path_large_targets_several_points(ard, mean_name, policy, monkeypatch):
    # correlated points and large targets of mixed signs
    gp = make_model(ard, mean_name, 2)
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [1.0, 1.0]])
    t = np.array([1e150, -1e150, 3e149, -2e149])
    mean_params = [1e149] if mean_name == "const" else []
    width_params = [0.0, 0.0] if ard else [0.0]
    params = np.array([np.log(1e-3)] + mean_params + width_params + [0.0])
    with np.errstate(**policy):
        (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    for fast, gen in ((f_fast, f_gen), (g_fast, g_gen)):
        if isinstance(gen, type):
            assert fast is gen
        else:
            numpy.testing.assert_allclose(fast, gen, rtol=1e-10)


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("log_scale", [0.0, np.log(1e-150)])
def test_fast_path_underflow_follows_generic(ard, log_scale, monkeypatch):
    # tiny products in the trace contractions: einsum, as in the generic
    # path, does not raise under np.errstate(under="raise")
    gp = make_model(ard, "zero", 1)
    X = np.array([[0.0], [np.sqrt(700.0)]])
    t = np.zeros(2)
    params = np.array([np.log(1e6), 0.0, log_scale])
    with np.errstate(under="raise"):
        (f_fast, g_fast), (f_gen, g_gen) = compare_both_paths(gp, X, t, params, monkeypatch)
    for fast, gen in ((f_fast, f_gen), (g_fast, g_gen)):
        if isinstance(gen, type):
            assert fast is gen
        else:
            numpy.testing.assert_allclose(fast, gen, rtol=1e-12)


def run_with_warnings(fn):
    """Result (or exception type) and the set of warnings (category, message)."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn()
        except Exception as e:  # compared by the caller
            result = type(e)
    return result, sorted({(w.category.__name__, str(w.message)) for w in caught})


@pytest.mark.parametrize(
    "mean_name, X, t, params",
    [
        # alpha ~ 3.4e307: the noise term overflows in both paths
        ("zero", [[0.0]], [1.7e308], [np.log(2.0), 0.0, 0.0]),
        # alpha ~ 1e308 twice: the mean term overflows in both paths
        ("const", [[0.0], [100.0]], [1e308, 1e308], [0.0, 0.0, 0.0, np.log(1e-6)]),
        # coordinates whose ratio to the width overflows
        ("zero", [[0.0], [1e308]], [0.0, 1.0], [0.0, np.log(1e-6), 0.0]),
        # non-finite targets and coordinates
        ("const", [[0.0], [0.5], [1.0]], [0.0, np.inf, 0.3], [np.log(0.1), 0.0, 0.0, 0.0]),
        ("const", [[0.0], [0.5], [1.0]], [0.0, np.nan, 0.3], [np.log(0.1), 0.0, 0.0, 0.0]),
        ("const", [[0.0], [np.nan], [1.0]], [0.0, 0.2, 0.3], [np.log(0.1), 0.0, 0.0, 0.0]),
        # no data
        ("zero", np.zeros((0, 1)), np.zeros(0), [0.0, 0.0, 0.0]),
        ("const", np.zeros((0, 1)), np.zeros(0), [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_fast_path_same_results_and_warnings(mean_name, X, t, params, monkeypatch):
    gp = make_model(False, mean_name, 1)
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    params = np.asarray(params)
    for method in ("eval_marlik", "get_grad_marlik"):
        fast = run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t))
        gen = generic(
            lambda: run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t)),
            monkeypatch,
        )
        assert fast[1] == gen[1]
        if isinstance(gen[0], type):
            assert fast[0] is gen[0]
        else:
            numpy.testing.assert_array_equal(fast[0], gen[0])


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
def test_fast_path_float32_params(ard, mean_name, monkeypatch):
    # the noise variance is promoted to float64 after squaring, as in the
    # generic path (2 * var overflows in float32)
    gp = make_model(ard, mean_name, 1)
    X = np.array([[0.0], [0.5]])
    t = np.array([0.0, 0.3])
    mean_params = [0.0] if mean_name == "const" else []
    params = np.array([44.1] + mean_params + [0.0, 0.0], dtype=np.float32)
    for method in ("eval_marlik", "get_grad_marlik"):
        fast = run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t))
        gen = generic(
            lambda: run_with_warnings(lambda: getattr(gp, method)(params.copy(), X, t)),
            monkeypatch,
        )
        assert fast[1] == gen[1]
        numpy.testing.assert_allclose(fast[0], gen[0], rtol=1e-6)


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("mean_name", ["const", "zero"])
def test_fast_path_integer_inputs(ard, mean_name, monkeypatch):
    gp = make_model(ard, mean_name, 2)
    X = np.array([[0, 1], [2, 3], [5, 1], [4, 4]])
    t = np.array([1, -2, 0, 3])
    np.random.seed(0)
    params = gp.get_cand_params(X.astype(float), t.astype(float))
    p_fast, p_gen = params.copy(), params.copy()
    f_fast = gp.eval_marlik(p_fast, X, t)
    g_fast = gp.get_grad_marlik(p_fast, X, t)
    f_gen = generic(lambda: gp.eval_marlik(p_gen, X, t), monkeypatch)
    g_gen = generic(lambda: gp.get_grad_marlik(p_gen, X, t), monkeypatch)
    assert f_fast == pytest.approx(f_gen, rel=1e-12)
    numpy.testing.assert_allclose(g_fast, g_gen, rtol=1e-10, atol=1e-12)
    numpy.testing.assert_array_equal(p_fast, p_gen)
