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
  must be left out by the caller.
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

_COMMON_PROPERTIES: dict[str, Any] = {
    "candidates": {
        "type": "array",
        "items": {"type": "array", "items": {"type": "number"}, "minItems": 1},
        "description": "Finite candidate set: one numeric feature vector per row. Exclusive with 'space'.",
    },
    "space": {
        "type": "array",
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
                "x": {"type": "array", "items": {"type": "number"}},
                "value": {"type": "number"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        "description": "Valid evaluations so far, each with 'index' or 'x' (exact match) and a finite 'value'.",
    },
    "direction": {"enum": ["maximize", "minimize"], "default": "maximize"},
    "seed": {"type": "integer", "default": 0},
    "scale_features": {"type": "boolean", "default": True},
    "standardize": {"type": "boolean", "default": True},
    "num_rand_basis": {
        "type": "integer",
        "minimum": 0,
        "default": 0,
        "description": "0: exact Gaussian process; >0: random-feature (Bayesian linear model) approximation.",
    },
}


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
            "additionalProperties": False,
        },
        "predict_request": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": dict(_COMMON_PROPERTIES),
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
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def _candidate_matrix(request: dict[str, Any]) -> tuple[np.ndarray, list[str] | None]:
    if "space" in request:
        axes = request["space"]
        names = [str(a["name"]) for a in axes]
        rows = list(itertools.product(*[a["values"] for a in axes]))
        return np.array(rows, dtype=float).reshape(len(rows), len(axes)), names
    return np.array(request["candidates"], dtype=float), None


def _n_candidates(request: dict[str, Any]) -> int:
    if "space" in request:
        return int(np.prod([len(a.get("values", [])) for a in request["space"]]))
    return len(request.get("candidates", []))


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
                or "name" not in ax
                or not isinstance(ax.get("values"), list)
                or not ax["values"]
            ):
                issues.append(
                    _issue(
                        "SPACE_AXIS",
                        f"axis {i} needs 'name' and non-empty 'values'",
                        f"space[{i}]",
                    )
                )
                return issues
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
                f"{n} candidates > {MAX_CANDIDATES}",
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
    for i, o in enumerate(obs):
        f = f"observations[{i}]"
        if not isinstance(o, dict) or not _is_num(o.get("value")):
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
            issues.append(
                _issue("OBSERVATION_REF", "give exactly one of 'index' or 'x'", f)
            )
        elif "index" in o:
            if (
                not isinstance(o["index"], int)
                or isinstance(o["index"], bool)
                or not 0 <= o["index"] < n
            ):
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
            issues.append(
                _issue("OBSERVATION_X", f"'x' must be {dim} finite numbers", f)
            )

    if request.get("direction", "maximize") not in ("maximize", "minimize"):
        issues.append(
            _issue(
                "INVALID_OPTION",
                "direction must be 'maximize' or 'minimize'",
                "direction",
            )
        )
    for key in ("seed", "num_rand_basis", "num_suggestions", "num_random_initial"):
        if key in request and (
            not isinstance(request[key], int) or isinstance(request[key], bool)
        ):
            issues.append(_issue("INVALID_OPTION", f"'{key}' must be an integer", key))
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


def _prepare(
    request: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str] | None, np.ndarray, np.ndarray]:
    """Returns (raw X, scaled X, names, observed indices, observed values in PHYSBO's maximise sense)."""
    X, names = _candidate_matrix(request)
    if request.get("scale_features", True):
        lo, hi = X.min(axis=0), X.max(axis=0)
        Xs = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    else:
        Xs = X
    lookup = {tuple(row): i for i, row in enumerate(X.tolist())}
    idx, val = [], []
    for o in request.get("observations", []):
        if "index" in o:
            i = int(o["index"])
        else:
            key = tuple(float(v) for v in o["x"])
            if key not in lookup:
                raise ValueError(f"observation x={o['x']} is not one of the candidates")
            i = lookup[key]
        idx.append(i)
        val.append(float(o["value"]))
    idx_a = np.array(idx, dtype=int)
    if len(set(idx)) != len(idx):
        raise ValueError(
            "duplicate observations of the same candidate; aggregate them before calling"
        )
    t = np.array(val, dtype=float)
    if request.get("direction", "maximize") == "minimize":
        t = -t
    if request.get("standardize", True) and len(t) > 1 and t.std() > 0:
        t = (t - t.mean()) / t.std()
    elif request.get("standardize", True) and len(t) > 0:
        t = t - t.mean()
    return X, Xs, names, idx_a, t


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
        X, Xs, names, idx, t = _prepare(request)
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
    with _isolated_rng(seed), contextlib.redirect_stdout(log):
        policy = physbo.search.discrete.Policy(
            test_X=Xs, initial_data=(idx, t) if len(idx) else None
        )
        policy.set_seed(seed)
        if len(idx) < max(n_init, 1):
            actions = policy.random_search(
                max_num_probes=1, num_search_each_probe=k, simulator=None, is_disp=False
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
    return {
        "status": "ok",
        "mode": mode,
        "suggestions": [_row(X, names, int(a)) for a in np.atleast_1d(actions)],
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
        X, Xs, _names, idx, t = _prepare(request)
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
    raw = np.array([float(o["value"]) for o in request["observations"]])
    sign = -1.0 if request.get("direction", "maximize") == "minimize" else 1.0
    signed = sign * raw
    if request.get("standardize", True):
        mu, sd = signed.mean(), (signed.std() if signed.std() > 0 else 1.0)
    else:
        mu, sd = 0.0, 1.0
    seed = int(request.get("seed", 0))
    with _isolated_rng(seed), contextlib.redirect_stdout(io.StringIO()):
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
        mean = policy.get_post_fmean(Xs)
        var = policy.get_post_fcov(Xs)
    mean = sign * (mean * sd + mu)
    std = np.sqrt(np.clip(var, 0.0, None)) * sd
    return {
        "status": "ok",
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_candidates": int(X.shape[0]),
        "n_observed": len(idx),
        "provenance": _provenance(request),
    }
