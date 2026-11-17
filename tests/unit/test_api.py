# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import json

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")
api = pytest.importorskip("physbo.api")


def branin(x, y):
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return float((y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * np.cos(x) + 10)


GX = [float(v) for v in np.linspace(-5, 10, 15)]
GY = [float(v) for v in np.linspace(0, 15, 15)]
SPACE = [{"name": "x", "values": GX}, {"name": "y", "values": GY}]
F = np.array([branin(x, y) for x in GX for y in GY])


def _loop(budget, seed=0, **extra):
    obs = []
    while len(obs) < budget:
        r = api.suggest(
            {
                "space": SPACE,
                "direction": "minimize",
                "seed": seed + len(obs),
                "observations": obs,
                **extra,
            }
        )
        assert r["status"] == "ok", r
        for s in r["suggestions"]:
            obs.append({"index": s["index"], "value": F[s["index"]]})
    return obs


def test_schema_is_json_and_versioned():
    s = api.schema()
    json.dumps(s)
    assert s["schema_version"] == api.SCHEMA_VERSION
    assert "space" in s["suggest_request"]["properties"]


def test_validate_reports_structured_issues():
    issues = api.validate({"candidates": [[1.0, 2.0], [3.0]], "foo": 1})
    codes = {i["code"] for i in issues}
    assert {"UNKNOWN_FIELD", "CANDIDATES_SHAPE"} <= codes
    assert all(set(i) == {"code", "message", "field", "hint"} for i in issues)
    assert api.validate({"space": SPACE}) == []
    assert {
        i["code"] for i in api.validate({"space": SPACE, "candidates": [[0.0, 0.0]]})
    } == {"CANDIDATES_REQUIRED"}


def test_nan_and_bad_references_are_rejected():
    r = api.suggest(
        {"space": SPACE, "observations": [{"index": 0, "value": float("nan")}]}
    )
    assert r["status"] == "failed" and r["issues"][0]["code"] == "OBSERVATION_VALUE"
    r = api.suggest(
        {"space": SPACE, "observations": [{"x": [123.0, 0.0], "value": 1.0}]}
    )
    assert r["status"] == "failed" and r["issues"][0]["code"] == "OBSERVATION_REF"
    r = api.suggest(
        {
            "space": SPACE,
            "observations": [{"index": 3, "value": 1.0}, {"index": 3, "value": 2.0}],
        }
    )
    assert r["status"] == "failed"


def test_random_phase_then_bayes_and_no_repeats():
    r0 = api.suggest({"space": SPACE, "num_suggestions": 3})
    assert r0["mode"] == "random" and len(r0["suggestions"]) == 3
    assert set(r0["suggestions"][0]) == {"index", "x", "params"}
    obs = _loop(20)
    assert len({o["index"] for o in obs}) == len(
        obs
    )  # never suggests an observed candidate
    r = api.suggest({"space": SPACE, "direction": "minimize", "observations": obs})
    assert r["mode"] == "bayes" and r["suggestions"][0]["index"] not in {
        o["index"] for o in obs
    }


def test_deterministic_and_restores_global_rng():
    np.random.seed(7)
    expected = np.random.rand()
    np.random.seed(7)
    a = _loop(10, seed=1)
    assert np.random.rand() == expected
    b = _loop(10, seed=1)
    assert [o["index"] for o in a] == [o["index"] for o in b]


def test_beats_random_sampling_on_branin_grid():
    # 225 candidates, 25 evaluations: uniform random sampling has a median best-rank of about 4.
    vals = np.sort(F)
    ranks = [
        int((vals < min(o["value"] for o in _loop(25, seed=s))).sum()) for s in range(3)
    ]
    assert max(ranks) <= 3, ranks


def test_exhausting_the_candidate_set():
    small = [{"name": "x", "values": [0.0, 1.0, 2.0]}]
    obs = [{"index": i, "value": float(i)} for i in range(3)]
    r = api.suggest({"space": small, "observations": obs})
    assert r["status"] == "ok" and r["suggestions"] == [] and r["n_remaining"] == 0


def test_predict_returns_callers_units_and_sign():
    obs = _loop(15)
    p = api.predict({"space": SPACE, "direction": "minimize", "observations": obs})
    assert p["status"] == "ok" and len(p["mean"]) == len(F) == len(p["std"])
    observed = np.array([o["value"] for o in obs])
    pred = np.array([p["mean"][o["index"]] for o in obs])
    assert np.corrcoef(observed, pred)[0, 1] > 0.9  # same sign and scale as the inputs
    assert (
        api.predict({"space": SPACE, "observations": obs[:1]})["issues"][0]["code"]
        == "TOO_FEW_OBSERVATIONS"
    )


def test_mcp_server_tools_wrap_the_api():
    pytest.importorskip("mcp")
    from physbo import mcp_server

    server = mcp_server.build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "get_schema",
        "versions",
        "validate_request",
        "suggest_next",
        "predict_posterior",
    }

    def call(name, args):
        out = asyncio.run(server.call_tool(name, args))
        payload = out[1] if isinstance(out, tuple) else out
        if isinstance(payload, dict) and "result" in payload and len(payload) == 1:
            payload = payload["result"]
        if not isinstance(payload, dict):  # content blocks
            payload = json.loads(payload[0].text)
        return payload

    assert call("validate_request", {"request": {"space": SPACE}})["valid"] is True
    r = call("suggest_next", {"request": {"space": SPACE, "seed": 0}})
    assert r == api.suggest({"space": SPACE, "seed": 0})
