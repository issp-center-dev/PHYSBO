# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the generator of the benchmark-function catalog in the manual.

The catalog (docs/sphinx/manual/test_function_catalog.py) derives the
properties of the test functions from the classes; these tests make sure
that it covers every class, that its hand-written entries refer to existing
classes, and that the generated reStructuredText is well-formed.
"""

import importlib.util
import inspect
import pathlib

import pytest

physbo = pytest.importorskip("physbo")
docutils_core = pytest.importorskip("docutils.core")

from physbo.test_functions import base, multi_objective, single_objective

CATALOG_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs"
    / "sphinx"
    / "manual"
    / "test_function_catalog.py"
)


@pytest.fixture(scope="module")
def catalog():
    spec = importlib.util.spec_from_file_location("test_function_catalog", CATALOG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_classes(module, root):
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and issubclass(obj, root) and obj is not root
    }


@pytest.mark.parametrize(
    "module, root",
    [
        (multi_objective, multi_objective.MultiTestFunction),
        (single_objective, single_objective.SingleTestFunction),
    ],
    ids=["multi", "single"],
)
def test_catalog_covers_every_class(catalog, module, root):
    listed = set()
    for primary, aliases in catalog.collect(module, root):
        listed.add(primary.__name__)
        assert primary.__bases__[0] is root
        for alias in aliases:
            assert issubclass(alias, primary)
            listed.add(alias.__name__)
    assert listed == _all_classes(module, root)


def test_hand_written_entries_refer_to_existing_classes(catalog):
    multi = _all_classes(multi_objective, multi_objective.MultiTestFunction)
    assert set(catalog.PARETO_SET) <= multi
    assert set(catalog.SPECIAL) <= multi | _all_classes(
        single_objective, single_objective.SingleTestFunction
    )


def test_constraint_column_matches_override(catalog):
    for primary, _ in catalog.collect(multi_objective, multi_objective.MultiTestFunction):
        expected = "constraint" in vars(primary)
        assert catalog._has_constraint(primary) == expected, primary.__name__


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_generated_tables_are_valid_rst(catalog, lang, tmp_path):
    paths = catalog.write_tables(str(tmp_path), lang)
    assert len(paths) == 2
    for path in paths:
        text = pathlib.Path(path).read_text(encoding="utf-8")
        assert text.startswith(".. list-table::")
        # docutils reports malformed tables as system messages of level >= 2
        messages = []
        docutils_core.publish_doctree(
            text,
            settings_overrides={
                "report_level": 5,
                "halt_level": 5,
                "warning_stream": _Collector(messages),
            },
        )
        assert messages == [], "\n".join(messages)

    # every primary class appears in its table
    multi = pathlib.Path(paths[0]).read_text(encoding="utf-8")
    single = pathlib.Path(paths[1]).read_text(encoding="utf-8")
    for primary, aliases in catalog.collect(multi_objective, multi_objective.MultiTestFunction):
        assert f".{primary.__name__}`" in multi
        for alias in aliases:
            assert f".{alias.__name__}`" in multi
    for primary, _ in catalog.collect(single_objective, single_objective.SingleTestFunction):
        assert f".{primary.__name__}`" in single


class _Collector:
    """Minimal stream collecting docutils system messages."""

    def __init__(self, sink):
        self._sink = sink

    def write(self, text):
        if text.strip():
            self._sink.append(text.strip())

    def flush(self):
        pass


def test_format_domain(catalog):
    fn = multi_objective.BinhKorn()
    assert catalog.format_domain(fn, False) == r"[0, 5] \times [0, 3]"
    fn = multi_objective.ZDT4()
    assert catalog.format_domain(fn, True) == r"x_1 \in [0, 1],\ x_i \in [-5, 5]\ (i \ge 2)"
    fn = multi_objective.ZDT1()
    assert catalog.format_domain(fn, True) == "[0, 1]^{N}"
    fn = multi_objective.Schaffer1()
    assert catalog.format_domain(fn, False) == "[-10, 10]"
    fn = multi_objective.Poloni()
    assert catalog.format_domain(fn, False) == r"[-\pi, \pi]^{2}"
    assert base.TestFunction.constraint is not multi_objective.SRN.constraint
