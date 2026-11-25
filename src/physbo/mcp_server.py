# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MCP (Model Context Protocol) server for PHYSBO: a thin wrapper around :mod:`physbo.api`.

Run with ``physbo-mcp`` (stdio transport) after ``pip install "physbo[mcp]"``.

All tools are stateless and side-effect free: they take candidates and observations as JSON, return
JSON, write no files, open no network connections and never execute code supplied by the client.
PHYSBO proposes *where* to evaluate next; running the actual simulation or experiment stays with the
caller (ask-tell loop), so there is intentionally no "run" tool.
"""

from __future__ import annotations

from typing import Any

from . import api

INSTRUCTIONS = """\
PHYSBO: Bayesian optimisation over a finite candidate set (ask-tell).
1. get_schema() once to see the request format.
2. validate_request(request) to check a request without running anything.
3. suggest_next(request) -> candidates to evaluate next; evaluate them yourself (simulation/experiment).
4. Append each *valid* result to request['observations'] as {index, value} and call suggest_next again.
   Leave failed or non-converged evaluations out (do not pass NaN or a huge penalty).
5. predict_posterior(request) for the surrogate's mean/std on every candidate.
PHYSBO maximises by default; set direction='minimize' for losses/energies.
"""


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            'the MCP server needs the optional extra: pip install "physbo[mcp]"'
        ) from exc

    server = FastMCP("physbo", instructions=INSTRUCTIONS)

    @server.tool()
    def get_schema() -> dict[str, Any]:
        """JSON Schemas of suggest/predict requests and results, plus the API schema version."""
        return api.schema()

    @server.tool()
    def versions() -> dict[str, str]:
        """Versions of PHYSBO, NumPy and SciPy (record them for reproducibility)."""
        return api.versions()

    @server.tool()
    def validate_request(
        request: dict[str, Any], kind: str = "suggest"
    ) -> dict[str, Any]:
        """Check a suggest (kind='suggest') or predict (kind='predict') request without running it.

        Returns {"valid": bool, "issues": [{code, message, field, hint}]}.
        """
        if kind not in ("suggest", "predict"):
            return {
                "valid": False,
                "issues": [
                    {
                        "code": "INVALID_OPTION",
                        "message": "kind must be 'suggest' or 'predict'",
                        "field": "kind",
                        "hint": None,
                    }
                ],
            }
        issues = api.validate(request, kind=kind)
        return {"valid": not issues, "issues": issues}

    @server.tool()
    def suggest_next(request: dict[str, Any]) -> dict[str, Any]:
        """Next candidate(s) to evaluate given the candidate set and the valid observations so far.

        Request: {candidates | space, observations?, direction?, num_suggestions?, score? (EI|PI|TS),
        num_random_initial?, seed?, scale_features?, standardize?, num_rand_basis?}.
        Result: {status, mode (random|bayes), suggestions: [{index, x, params?}], n_candidates,
        n_observed, n_remaining, provenance} or {status: failed, issues}.
        """
        return api.suggest(request)

    @server.tool()
    def predict_posterior(request: dict[str, Any]) -> dict[str, Any]:
        """Posterior mean and standard deviation on every candidate (needs >= 2 observations)."""
        return api.predict(request)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
