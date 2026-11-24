# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stateless, JSON-in / JSON-out "ask-tell" interface to discrete single-objective search.

This module is the shared core used by the MCP server (:mod:`physbo.mcp_server`) and by external
workflow tools (e.g. V-Lab Platform adapters). Every function takes and returns plain JSON-compatible
dictionaries, never executes user code, writes no files and restores the global NumPy RNG state.

Functions
---------
schema()            JSON Schemas of the requests and results (versioned)
validate(request)   check a request without running anything; returns a list of issues
suggest(request)    next candidate(s) to evaluate, given the candidate set and the observations so far
predict(request)    posterior mean / standard deviation on the candidates
versions()          PHYSBO / NumPy / SciPy versions (for provenance)

Conventions
-----------
* Candidates are a finite set. Give them either as ``candidates`` (list of numeric feature vectors)
  or as ``space`` (named axes with explicit ``values``; the candidate set is their Cartesian product
  in row-major order).
* ``observations`` refer to candidates by ``index`` (row of the candidate set) or by ``x`` (exact
  feature vector). Only *valid* evaluations should be passed; failed or non-converged evaluations
  must be left out by the caller. Each candidate may be observed at most once.
* ``direction`` is ``"maximize"`` (PHYSBO's native convention) or ``"minimize"``.
* By default, features are min-max scaled to [0, 1] per axis and observed values are standardised
  (zero mean, unit variance) before fitting the Gaussian process. Both can be switched off.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
SCORES = ("EI", "PI", "TS")
MAX_CANDIDATES = 1_000_000
MAX_RAND_BASIS = 20_000
SEED_MAX = 2**32 - 1  # np.random.seed accepts [0, 2**32 - 1]

_COMMON_PROPERTIES: dict[str, Any] = {
    "candidates": {
        "type": "array",
        "minItems": 1,
        "items": {"type": "array", "items": {"type": "number"}, "minItems": 1},
        "description": "Finite candidate set: one numeric feature vector per row. Exclusive with 'space'.",
    },
    "space": {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "array", "items": {"type": "number"}, "minItems": 1},
            },
            "required": ["name", "values"],
            "additionalProperties": False,
        },
        "description": "Named axes; the candidate set is the Cartesian product (row-major). Exclusive with 'candidates'.",
    },
    "observations": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "x": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                "value": {"type": "number"},
            },
            "required": ["value"],
            "oneOf": [{"required": ["index"]}, {"required": ["x"]}],
            "additionalProperties": False,
        },
        "description": "Valid evaluations so far, each with 'index' or 'x' (exact match) and a finite 'value'. "
        "Each candidate may be observed at most once.",
    },
    "direction": {"enum": ["maximize", "minimize"], "default": "maximize"},
    "seed": {"type": "integer", "minimum": 0, "maximum": SEED_MAX, "default": 0},
    "scale_features": {"type": "boolean", "default": True},
    "standardize": {"type": "boolean", "default": True},
    "num_rand_basis": {
        "type": "integer",
        "minimum": 0,
        "maximum": MAX_RAND_BASIS,
        "default": 0,
        "description": "0: exact Gaussian process; >0: random-feature (Bayesian linear model) approximation.",
    },
}


_ONE_CANDIDATE_SET = [{"required": ["candidates"]}, {"required": ["space"]}]


def schema() -> dict[str, Any]:
    """JSON Schemas (draft 2020-12) for the requests and results of this module."""
    suggest_props = dict(_COMMON_PROPERTIES)
    suggest_props.update(
        {
            "num_suggestions": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 1,
            },
            "score": {"enum": list(SCORES), "default": "EI"},
            "num_random_initial": {
                "type": "integer",
                "minimum": 0,
                "default": 5,
                "description": "While fewer valid observations exist, suggestions are uniformly random.",
            },
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "suggest_request": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": suggest_props,
            "oneOf": _ONE_CANDIDATE_SET,
            "additionalProperties": False,
        },
        "predict_request": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": dict(_COMMON_PROPERTIES),
            "oneOf": _ONE_CANDIDATE_SET,
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {
                "status": {"enum": ["ok", "failed"]},
                "issues": {"type": "array"},
                "mode": {"enum": ["random", "bayes"]},
                "suggestions": {"type": "array"},
                "n_candidates": {"type": "integer"},
                "n_observed": {"type": "integer"},
                "n_remaining": {"type": "integer"},
                "provenance": {"type": "object"},
            },
        },
    }


# ---------------------------------------------------------------------------------- helpers
def _issue(
    code: str, message: str, field: str | None = None, hint: str | None = None
) -> dict[str, Any]:
    return {"code": code, "message": message, "field": field, "hint": hint}


def _is_num(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:  # JSON integers beyond the float64 range
        return False


def _candidate_matrix(request: dict[str, Any]) -> tuple[np.ndarray, list[str] | None]:
    if "space" in request:
        axes = request["space"]
        names = [str(a["name"]) for a in axes]
        rows = list(itertools.product(*[a["values"] for a in axes]))
        return np.array(rows, dtype=float).reshape(len(rows), len(axes)), names
    return np.array(request["candidates"], dtype=float), None


def _n_candidates(request: dict[str, Any]) -> int:
    """Number of candidates, in exact integer arithmetic; stops counting once MAX_CANDIDATES is exceeded."""
    if "space" in request:
        n = 1
        for a in request["space"]:
            n *= len(a["values"])
            if n > MAX_CANDIDATES:
                return n
        return n
    return len(request["candidates"])


def _resolve_observations(
    request: dict[str, Any], n: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Row index of every observation, plus issues for unresolved ``x`` and duplicate candidates.

    Requires a request whose candidate set and observation entries are already well formed. An ``x``
    that appears several times in the candidate set resolves to its first row.
    """
    issues: list[dict[str, Any]] = []
    obs = request.get("observations", [])
    lookup: dict[tuple[float, ...], int] | None = None
    first: list[dict[float, int]] | None = None
    if any("x" in o for o in obs):
        if "space" in request:
            first = []
            for ax in request["space"]:
                d: dict[float, int] = {}
                for j, v in enumerate(ax["values"]):
                    d.setdefault(float(v), j)
                first.append(d)
        else:
            lookup = {}
            for i, row in enumerate(request["candidates"]):
                lookup.setdefault(tuple(float(v) for v in row), i)
    idx: list[int] = []
    seen: dict[int, int] = {}
    for k, o in enumerate(obs):
        f = f"observations[{k}]"
        if "index" in o:
            i = int(o["index"])
        else:
            key = tuple(float(v) for v in o["x"])
            i = -1
            if lookup is not None:
                i = lookup.get(key, -1)
            elif first is not None:
                i = 0
                for d, ax, v in zip(first, request["space"], key):
                    j = d.get(v)
                    if j is None:
                        i = -1
                        break
                    i = i * len(ax["values"]) + j
            if i < 0:
                issues.append(
                    _issue(
                        "OBSERVATION_REF",
                        f"observation x={o['x']} is not one of the candidates",
                        f,
                    )
                )
                continue
        if i in seen:
            issues.append(
                _issue(
                    "OBSERVATION_DUPLICATE",
                    f"candidate {i} is also observed in observations[{seen[i]}]",
                    f,
                    "aggregate repeated evaluations of a candidate before calling",
                )
            )
            continue
        seen[i] = k
        idx.append(i)
    return idx, issues


def versions() -> dict[str, str]:
    """Versions of PHYSBO and its numerical dependencies."""
    import scipy

    import physbo

    return {
        "physbo": physbo.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def validate(request: dict[str, Any], *, kind: str = "suggest") -> list[dict[str, Any]]:
    """Check a request without running anything. ``kind`` is ``"suggest"`` or ``"predict"``.

    Returns a list of issues ``{code, message, field, hint}``; an empty list means the request is valid.
    """
    issues: list[dict[str, Any]] = []
    if kind not in ("suggest", "predict"):
        return [_issue("INVALID_OPTION", "kind must be 'suggest' or 'predict'", "kind")]
    if not isinstance(request, dict):
        return [_issue("INVALID_REQUEST", "request must be a JSON object")]
    allowed = set(schema()[f"{kind}_request"]["properties"])
    for key in sorted(set(request) - allowed):
        issues.append(_issue("UNKNOWN_FIELD", f"unknown field '{key}'", key))

    has_c, has_s = "candidates" in request, "space" in request
    if has_c == has_s:
        issues.append(
            _issue("CANDIDATES_REQUIRED", "give exactly one of 'candidates' or 'space'")
        )
        return issues
    dim = None
    if has_c:
        c = request["candidates"]
        if not isinstance(c, list) or not c:
            issues.append(
                _issue(
                    "CANDIDATES_REQUIRED",
                    "'candidates' must be a non-empty list",
                    "candidates",
                )
            )
            return issues
        dims = {len(r) if isinstance(r, list) else -1 for r in c}
        if len(dims) != 1 or -1 in dims or 0 in dims:
            issues.append(
                _issue(
                    "CANDIDATES_SHAPE",
                    "every candidate must be a non-empty list of equal length",
                    "candidates",
                )
            )
            return issues
        if not all(_is_num(v) for r in c for v in r):
            issues.append(
                _issue(
                    "CANDIDATES_NOT_NUMERIC",
                    "candidate features must be finite numbers",
                    "candidates",
                    "encode categorical variables numerically (e.g. one-hot)",
                )
            )
        dim = dims.pop()
    else:
        s = request["space"]
        if not isinstance(s, list) or not s:
            issues.append(
                _issue(
                    "CANDIDATES_REQUIRED",
                    "'space' must be a non-empty list of axes",
                    "space",
                )
            )
            return issues
        names = []
        for i, ax in enumerate(s):
            if (
                not isinstance(ax, dict)
                or not isinstance(ax.get("name"), str)
                or not isinstance(ax.get("values"), list)
                or not ax["values"]
            ):
                issues.append(
                    _issue(
                        "SPACE_AXIS",
                        f"axis {i} needs a string 'name' and non-empty 'values'",
                        f"space[{i}]",
                    )
                )
                return issues
            for key in sorted(set(ax) - {"name", "values"}):
                issues.append(
                    _issue(
                        "UNKNOWN_FIELD", f"unknown field '{key}'", f"space[{i}].{key}"
                    )
                )
            if not all(_is_num(v) for v in ax["values"]):
                issues.append(
                    _issue(
                        "CANDIDATES_NOT_NUMERIC",
                        f"axis '{ax['name']}' values must be finite numbers",
                        f"space[{i}]",
                    )
                )
            names.append(ax["name"])
        if len(set(names)) != len(names):
            issues.append(_issue("SPACE_AXIS", "axis names must be unique", "space"))
        dim = len(s)
    n = _n_candidates(request)
    if n > MAX_CANDIDATES:
        issues.append(
            _issue(
                "TOO_MANY_CANDIDATES",
                f"more than {MAX_CANDIDATES} candidates",
                None,
                "reduce the grid or use the continuous (range) search",
            )
        )

    obs = request.get("observations", [])
    if not isinstance(obs, list):
        issues.append(
            _issue(
                "OBSERVATIONS_INVALID", "'observations' must be a list", "observations"
            )
        )
        obs = []
    obs_ok = not issues and n <= MAX_CANDIDATES
    for i, o in enumerate(obs):
        f = f"observations[{i}]"
        if isinstance(o, dict):
            for key in sorted(set(o) - {"index", "x", "value"}):
                issues.append(
                    _issue("UNKNOWN_FIELD", f"unknown field '{key}'", f"{f}.{key}")
                )
                obs_ok = False
        if not isinstance(o, dict) or not _is_num(o.get("value")):
            obs_ok = False
            issues.append(
                _issue(
                    "OBSERVATION_VALUE",
                    "each observation needs a finite numeric 'value'",
                    f,
                    "leave failed / non-converged evaluations out instead of passing NaN",
                )
            )
            continue
        if ("index" in o) == ("x" in o):
            obs_ok = False
            issues.append(
                _issue("OBSERVATION_REF", "give exactly one of 'index' or 'x'", f)
            )
        elif "index" in o:
            if (
                not isinstance(o["index"], int)
                or isinstance(o["index"], bool)
                or not 0 <= o["index"] < n
            ):
                obs_ok = False
                issues.append(
                    _issue(
                        "OBSERVATION_INDEX", f"index must be an integer in [0, {n})", f
                    )
                )
        elif (
            not isinstance(o["x"], list)
            or len(o["x"]) != dim
            or not all(_is_num(v) for v in o["x"])
        ):
            obs_ok = False
            issues.append(
                _issue("OBSERVATION_X", f"'x' must be {dim} finite numbers", f)
            )
    if obs_ok:
        issues.extend(_resolve_observations(request, n)[1])

    if request.get("direction", "maximize") not in ("maximize", "minimize"):
        issues.append(
            _issue(
                "INVALID_OPTION",
                "direction must be 'maximize' or 'minimize'",
                "direction",
            )
        )
    bounds = {
        "seed": (0, SEED_MAX),
        "num_rand_basis": (0, MAX_RAND_BASIS),
        "num_random_initial": (0, None),
    }
    for key in ("seed", "num_rand_basis", "num_suggestions", "num_random_initial"):
        if key not in request:
            continue
        v = request[key]
        if not isinstance(v, int) or isinstance(v, bool):
            issues.append(_issue("INVALID_OPTION", f"'{key}' must be an integer", key))
        elif key in bounds:
            lo, hi = bounds[key]
            if v < lo or (hi is not None and v > hi):
                rng = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                issues.append(
                    _issue("INVALID_OPTION", f"'{key}' must be {rng}", key)
                )
    if kind == "suggest":
        if request.get("score", "EI") not in SCORES:
            issues.append(
                _issue(
                    "INVALID_OPTION", f"score must be one of {list(SCORES)}", "score"
                )
            )
        k = request.get("num_suggestions", 1)
        if isinstance(k, int) and not 1 <= k <= 1000:
            issues.append(
                _issue(
                    "INVALID_OPTION",
                    "num_suggestions must be in [1, 1000]",
                    "num_suggestions",
                )
            )
    for key in ("scale_features", "standardize"):
        if key in request and not isinstance(request[key], bool):
            issues.append(_issue("INVALID_OPTION", f"'{key}' must be a boolean", key))
    return issues


class _NumericalError(ValueError):
    """The inputs cannot be represented in floating point after preprocessing."""


def _prepare(
    request: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str] | None, np.ndarray, np.ndarray, float, float]:
    """Raw X, scaled X, names, observed indices, preprocessed targets (PHYSBO's maximise sense), and
    the offset / scale that map preprocessed targets back to signed values (``signed = t * sd + mu``).

    Requires a request that :func:`validate` accepted. Scaling and standardisation are computed so
    that finite inputs cannot overflow; a non-finite result raises :class:`_NumericalError`.
    """
    X, names = _candidate_matrix(request)
    if request.get("scale_features", True):
        lo, hi = X.min(axis=0), X.max(axis=0)
        with np.errstate(over="ignore"):
            span = hi - lo
            wide = ~np.isfinite(span)
            Xs = (X - lo) / np.where(hi > lo, np.where(wide, 1.0, span), 1.0)
        if wide.any():
            # The span exceeds float64: halve first. Halving may round subnormal values, but such
            # differences are far below the resolution of a column that spans ~1e308.
            h_lo, h_hi = lo[wide] / 2, hi[wide] / 2
            Xs[:, wide] = (X[:, wide] / 2 - h_lo) / (h_hi - h_lo)
    else:
        Xs = X
    idx, ref_issues = _resolve_observations(request, X.shape[0])
    if ref_issues:  # validate() reports these; kept as a guard
        raise ValueError(ref_issues[0]["message"])
    t = np.array([float(o["value"]) for o in request.get("observations", [])])
    if request.get("direction", "maximize") == "minimize":
        t = -t
    mu, sd = 0.0, 1.0
    if request.get("standardize", True) and len(t) > 0:
        m = float(np.max(np.abs(t)))
        u = t / m if m > 0 else t
        u_mean, u_std = float(u.mean()), float(u.std())
        if len(t) > 1 and u_std > 0:
            t = (u - u_mean) / u_std
            mu, sd = m * u_mean, m * u_std
            if not (sd > 0 and math.isfinite(sd)):  # e.g. subnormal values: scale underflows
                raise _NumericalError(
                    "the spread of the observed values is not representable in floating point"
                )
        else:
            t = (u - u_mean) * (m if m > 0 else 1.0)
            mu = m * u_mean if m > 0 else u_mean
    if not (np.isfinite(Xs).all() and np.isfinite(t).all()):
        raise _NumericalError(
            "features or values overflow floating point after scaling"
        )
    return X, Xs, names, np.array(idx, dtype=int), t, mu, sd


def _numerical_failure(message: str) -> dict[str, Any]:
    return _failed(
        [
            _issue(
                "NUMERICAL_FAILURE",
                message,
                None,
                "rescale features / values to a moderate range (e.g. O(1)) and retry",
            )
        ]
    )


def _row(X: np.ndarray, names: list[str] | None, i: int) -> dict[str, Any]:
    out: dict[str, Any] = {"index": int(i), "x": [float(v) for v in X[i]]}
    if names is not None:
        out["params"] = {n: float(v) for n, v in zip(names, X[i])}
    return out


def _provenance(request: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "api_schema_version": SCHEMA_VERSION,
        "versions": versions(),
        "seed": int(request.get("seed", 0)),
        **extra,
    }


@contextlib.contextmanager
def _isolated_rng(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _failed(issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "failed", "issues": issues}


def suggest(request: dict[str, Any]) -> dict[str, Any]:
    """Next candidate(s) to evaluate. Never returns an already-observed candidate.

    Returns ``{"status": "ok", "mode": "random"|"bayes", "suggestions": [{index, x, params?}], ...}`` or
    ``{"status": "failed", "issues": [...]}``.
    """
    import physbo

    issues = validate(request, kind="suggest")
    if issues:
        return _failed(issues)
    try:
        X, Xs, names, idx, t, _mu, _sd = _prepare(request)
    except _NumericalError as exc:
        return _numerical_failure(str(exc))
    except ValueError as exc:
        return _failed([_issue("OBSERVATION_REF", str(exc), "observations")])
    n = X.shape[0]
    k = min(int(request.get("num_suggestions", 1)), n - len(idx))
    base = {"n_candidates": n, "n_observed": len(idx), "n_remaining": int(n - len(idx))}
    if k <= 0:
        return {
            "status": "ok",
            "mode": "random",
            "suggestions": [],
            **base,
            "message": "all candidates have been observed",
            "provenance": _provenance(request),
        }
    seed = int(request.get("seed", 0))
    score = str(request.get("score", "EI"))
    n_init = int(request.get("num_random_initial", 5))
    log = io.StringIO()
    try:
        with _isolated_rng(seed), contextlib.redirect_stdout(log), np.errstate(
            over="ignore", invalid="ignore"
        ):
            policy = physbo.search.discrete.Policy(
                test_X=Xs, initial_data=(idx, t) if len(idx) else None
            )
            policy.set_seed(seed)
            if len(idx) < max(n_init, 1):
                actions = policy.random_search(
                    max_num_probes=1,
                    num_search_each_probe=k,
                    simulator=None,
                    is_disp=False,
                )
                mode = "random"
            else:
                actions = policy.bayes_search(
                    max_num_probes=1,
                    num_search_each_probe=k,
                    simulator=None,
                    score=score,
                    interval=0,
                    num_rand_basis=int(request.get("num_rand_basis", 0)),
                    is_disp=False,
                )
                mode = "bayes"
            actions = np.atleast_1d(actions).astype(int)
            if mode == "bayes" and not np.isfinite(
                policy.get_post_fmean(Xs[actions])
            ).all():
                return _numerical_failure(
                    "the Gaussian process produced non-finite values"
                )
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        return _numerical_failure(f"{type(exc).__name__}: {exc}")
    observed = set(idx.tolist())
    if (
        len(set(actions.tolist())) != len(actions)
        or any(not 0 <= a < n or a in observed for a in actions.tolist())
    ):
        return _numerical_failure("PHYSBO returned an invalid suggestion")
    return {
        "status": "ok",
        "mode": mode,
        "suggestions": [_row(X, names, int(a)) for a in actions],
        **base,
        "provenance": _provenance(request, score=score if mode == "bayes" else None),
    }


def predict(request: dict[str, Any]) -> dict[str, Any]:
    """Posterior mean and standard deviation of the objective on every candidate.

    Values are returned in the caller's units and sign convention (standardisation is undone).
    Requires at least two observations.
    """
    import physbo

    issues = validate(request, kind="predict")
    if issues:
        return _failed(issues)
    try:
        X, Xs, _names, idx, t, mu, sd = _prepare(request)
    except _NumericalError as exc:
        return _numerical_failure(str(exc))
    except ValueError as exc:
        return _failed([_issue("OBSERVATION_REF", str(exc), "observations")])
    if len(idx) < 2:
        return _failed(
            [
                _issue(
                    "TOO_FEW_OBSERVATIONS",
                    "predict needs at least 2 observations",
                    "observations",
                )
            ]
        )
    sign = -1.0 if request.get("direction", "maximize") == "minimize" else 1.0
    seed = int(request.get("seed", 0))
    try:
        with _isolated_rng(seed), contextlib.redirect_stdout(
            io.StringIO()
        ), np.errstate(over="ignore", invalid="ignore"):
            policy = physbo.search.discrete.Policy(test_X=Xs, initial_data=(idx, t))
            policy.set_seed(seed)
            policy.bayes_search(
                max_num_probes=0,
                simulator=None,
                score="EI",
                interval=0,
                num_rand_basis=int(request.get("num_rand_basis", 0)),
                is_disp=False,
            )
            mean = sign * (policy.get_post_fmean(Xs) * sd + mu)
            std = np.sqrt(np.clip(policy.get_post_fcov(Xs), 0.0, None)) * sd
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        return _numerical_failure(f"{type(exc).__name__}: {exc}")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        return _numerical_failure(
            "the posterior is not finite in the caller's units"
        )
    return {
        "status": "ok",
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_candidates": int(X.shape[0]),
        "n_observed": len(idx),
        "provenance": _provenance(request),
    }
