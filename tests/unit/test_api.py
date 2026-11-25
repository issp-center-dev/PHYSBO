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


# ------------------------------------------------------------- review regressions
def _codes(issues):
    return {i["code"] for i in issues}


def test_candidate_count_does_not_wrap_around():
    space = [{"name": f"a{i}", "values": [0.0, 1.0]} for i in range(64)]
    assert _codes(api.validate({"space": space})) == {"TOO_MANY_CANDIDATES"}
    r = api.suggest({"space": space})
    assert r["status"] == "failed"
    assert r["issues"][0]["code"] == "TOO_MANY_CANDIDATES"


@pytest.mark.parametrize("name", [[], {}, 1, None])
def test_axis_names_must_be_strings(name):
    req = {"space": [{"name": name, "values": [0.0, 1.0]}]}
    assert _codes(api.validate(req)) == {"SPACE_AXIS"}
    assert api.suggest(req)["status"] == "failed"


def test_numeric_name_cannot_collide_with_string_name():
    req = {"space": [{"name": 1, "values": [0.0]}, {"name": "1", "values": [0.0]}]}
    assert "SPACE_AXIS" in _codes(api.validate(req))


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", -1),
        ("seed", 2**32),
        ("num_rand_basis", -1),
        ("num_rand_basis", api.MAX_RAND_BASIS + 1),
        ("num_random_initial", -1),
    ],
)
def test_integer_options_are_range_checked(key, value):
    req = {"space": SPACE, key: value}
    assert _codes(api.validate(req)) == {"INVALID_OPTION"}
    assert api.suggest(req)["issues"][0]["field"] == key
    if key != "num_random_initial":
        obs = [{"index": 0, "value": 1.0}, {"index": 1, "value": 2.0}]
        p = api.predict({"space": SPACE, "observations": obs, key: value})
        assert p["status"] == "failed" and p["issues"][0]["code"] == "INVALID_OPTION"


def test_seed_bounds_are_accepted():
    assert api.suggest({"space": SPACE, "seed": 2**32 - 1})["status"] == "ok"


def test_unresolved_and_duplicate_observations_fail_validation():
    bad_x = {"space": SPACE, "observations": [{"x": [123.0, 0.0], "value": 1.0}]}
    assert _codes(api.validate(bad_x)) == {"OBSERVATION_REF"}
    dup = {
        "space": SPACE,
        "observations": [
            {"index": 3, "value": 1.0},
            {"x": [GX[0], GY[3]], "value": 2.0},  # also row 3
        ],
    }
    assert _codes(api.validate(dup)) == {"OBSERVATION_DUPLICATE"}
    r = api.suggest(dup)
    assert r["status"] == "failed" and r["issues"][0]["code"] == "OBSERVATION_DUPLICATE"
    cands = {"candidates": [[0.0, 1.0], [2.0, 3.0]]}
    assert api.validate({**cands, "observations": [{"x": [2.0, 3.0], "value": 1}]}) == []
    assert _codes(
        api.validate({**cands, "observations": [{"x": [2.0, 4.0], "value": 1}]})
    ) == {"OBSERVATION_REF"}


def test_x_reference_resolves_to_the_same_row_as_index():
    req = {"space": SPACE, "observations": [{"x": [GX[2], GY[5]], "value": 1.0}]}
    r = api.suggest({**req, "num_suggestions": 224})
    assert 2 * len(GY) + 5 not in {s["index"] for s in r["suggestions"]}


def test_unknown_nested_fields_are_reported():
    req = {"space": [{"name": "x", "values": [0.0, 1.0], "unit": "eV"}]}
    assert _codes(api.validate(req)) == {"UNKNOWN_FIELD"}
    req = {"space": SPACE, "observations": [{"index": 0, "value": 1.0, "ok": True}]}
    assert _codes(api.validate(req)) == {"UNKNOWN_FIELD"}


def test_extreme_finite_inputs_give_json_or_a_structured_failure():
    req = {
        "candidates": [[-1e308], [0.0], [1e308]],
        "observations": [{"index": 0, "value": 1.0}, {"index": 1, "value": 2.0}],
    }
    for result in (api.predict(req), api.suggest({**req, "num_random_initial": 0})):
        json.dumps(result, allow_nan=False)
    p = api.predict(req)
    assert p["status"] == "ok"
    assert p["mean"][0] == pytest.approx(1.0, abs=0.1)
    big = {
        "candidates": [[0.0], [1.0], [2.0], [3.0]],
        "observations": [
            {"index": i, "value": v} for i, v in enumerate([1e308, -1e308, 1e308])
        ],
    }
    for result in (
        api.predict(big),
        api.predict({**big, "standardize": False}),
        api.suggest({**big, "num_random_initial": 0}),
        api.suggest({**big, "num_random_initial": 0, "standardize": False}),
    ):
        json.dumps(result, allow_nan=False)
        if result["status"] == "failed":
            assert result["issues"][0]["code"] == "NUMERICAL_FAILURE"


def test_predict_matches_a_direct_policy_fit():
    obs = _loop(12)
    req = {"space": SPACE, "direction": "minimize", "observations": obs, "seed": 4}
    p = api.predict(req)
    X = np.array([[x, y] for x in GX for y in GY])
    Xs = (X - X.min(0)) / (X.max(0) - X.min(0))
    t = -np.array([o["value"] for o in obs])
    mu, sd = t.mean(), t.std()
    state = np.random.get_state()
    np.random.seed(4)
    policy = physbo.search.discrete.Policy(
        test_X=Xs,
        initial_data=(np.array([o["index"] for o in obs]), (t - mu) / sd),
    )
    policy.set_seed(4)
    policy.bayes_search(
        max_num_probes=0, simulator=None, score="EI", interval=0, is_disp=False
    )
    mean = -(policy.get_post_fmean(Xs) * sd + mu)
    std = np.sqrt(policy.get_post_fcov(Xs)) * sd
    np.random.set_state(state)
    np.testing.assert_allclose(p["mean"], mean, rtol=1e-8, atol=1e-8 * sd)
    np.testing.assert_allclose(p["std"], std, rtol=1e-8, atol=1e-8 * sd)


def test_schema_states_exclusive_alternatives():
    s = api.schema()
    for kind in ("suggest_request", "predict_request"):
        assert s[kind]["oneOf"] == [
            {"required": ["candidates"]},
            {"required": ["space"]},
        ]
        props = s[kind]["properties"]
        assert props["candidates"]["minItems"] == 1 == props["space"]["minItems"]
        assert props["observations"]["items"]["oneOf"] == [
            {"required": ["index"]},
            {"required": ["x"]},
        ]
        assert props["seed"]["minimum"] == 0
        assert props["seed"]["maximum"] == 2**32 - 1


def test_schema_agrees_with_validate_when_jsonschema_is_available():
    jsonschema = pytest.importorskip("jsonschema")
    s = api.schema()["suggest_request"]
    cases = [
        {},
        {"space": SPACE, "candidates": [[0.0, 0.0]]},
        {"space": SPACE, "observations": [{"value": 1.0}]},
        {"space": SPACE, "observations": [{"index": 0, "x": [0.0, 0.0], "value": 1}]},
        {"space": SPACE, "seed": -1},
        {"space": SPACE},
    ]
    for req in cases:
        schema_ok = jsonschema.Draft202012Validator(s).is_valid(req)
        assert schema_ok == (api.validate(req) == []), req


def test_oversized_json_integers_are_structured_issues():
    huge = 10**400
    reqs = [
        {"candidates": [[huge]]},
        {"space": [{"name": "x", "values": [0.0, huge]}]},
        {"space": SPACE, "observations": [{"index": 0, "value": huge}]},
        {"space": SPACE, "observations": [{"x": [huge, 0.0], "value": 1.0}]},
        {"space": SPACE, "seed": huge},
        {"space": SPACE, "num_suggestions": huge},
    ]
    for req in reqs:
        assert api.validate(req), req
        for result in (api.suggest(req), api.predict(req)):
            assert result["status"] == "failed"
            json.dumps(result, allow_nan=False)


def test_feature_scaling_keeps_subnormal_and_huge_ranges():
    tiny = {
        "candidates": [[0.0], [5e-324], [1e-323]],
        "observations": [{"index": 0, "value": 0.0}, {"index": 2, "value": 1.0}],
    }
    _X, Xs, *_ = api._prepare(tiny)
    np.testing.assert_allclose(Xs[:, 0], [0.0, 0.5, 1.0])
    _X, Xs, *_ = api._prepare({"candidates": [[-1e308], [0.0], [1e308]]})
    np.testing.assert_allclose(Xs[:, 0], [0.0, 0.5, 1.0])
    _X, Xs, *_ = api._prepare({"candidates": [[-5e-324, 2.0], [5e-324, 4.0]]})
    np.testing.assert_allclose(Xs, [[0.0, 0.0], [1.0, 1.0]])


def test_validate_rejects_unknown_kind():
    assert _codes(api.validate({"space": SPACE}, kind="other")) == {"INVALID_OPTION"}


def test_subnormal_objective_spread_is_a_numerical_failure():
    req = {
        "candidates": [[0.0], [1.0], [2.0]],
        "observations": [{"index": 0, "value": 0.0}, {"index": 1, "value": 5e-324}],
    }
    for result in (api.predict(req), api.suggest({**req, "num_random_initial": 0})):
        assert result["status"] == "failed"
        assert result["issues"][0]["code"] == "NUMERICAL_FAILURE"
    # without standardisation the values are used as given; the result must still be strict JSON
    json.dumps(api.predict({**req, "standardize": False}), allow_nan=False)
