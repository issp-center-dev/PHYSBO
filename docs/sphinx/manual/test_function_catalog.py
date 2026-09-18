# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Generate the catalog tables of the bundled test functions.

The tables list the properties a user needs to choose a test function
(number of objectives and variables, constraints, sense, search space,
Pareto-optimal set / global minimum).  Everything that can be read from the
classes is generated from them, so that the tables cannot drift from the
implementation; only the Pareto-optimal sets are written by hand
(``PARETO_SET`` below).

Used by ``conf.py`` of each language (see :func:`sphinx_setup`) and by
``tests/unit/test_test_function_catalog.py``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import numpy as np

from physbo.test_functions import base, multi_objective, single_objective

# ---------------------------------------------------------------------------
# hand-written information
# ---------------------------------------------------------------------------

# Pareto-optimal set of the multi-objective functions (LaTeX, in terms of the
# decision variables), for the functions where a closed form is known.
PARETO_SET: dict[str, str] = {
    "Gaussian": r"\text{convex hull of the centers } \boldsymbol{c}_n \text{ (for } w_n = w \text{)}",
    "FonsecaFleming": r"x_1 = \cdots = x_N \in [-1/\sqrt{N}, 1/\sqrt{N}]",
    "BinhKorn": r"x_1 = x_2 \in [0, 3];\ x_1 \in [3, 5], x_2 = 3",
    "KitaYabumotoMoriNishikawa": r"x_1 \in [0, 3],\ x_2 = 13/2 - x_1/6",
    "Binh1": r"x_1 = x_2 \in [0, 5]",
    "Binh6": r"x_3 = x_4 = 1\ (x_1, x_2 \text{ arbitrary})",
    "Binh8": r"x_1 \in [0, 1],\ x_2 = 0",
    "Schaffer1": r"x \in [0, 2]",
    "Schaffer2": r"x \in [1, 2] \cup [4, 5]",
    "ZDT1": r"x_1 \in [0, 1],\ x_i = 0\ (i \ge 2)",
    "ZDT2": r"x_1 \in [0, 1],\ x_i = 0\ (i \ge 2)",
    "ZDT3": r"x_1 \in [0, 1] \text{ (disconnected)},\ x_i = 0\ (i \ge 2)",
    "ZDT4": r"x_1 \in [0, 1],\ x_i = 0\ (i \ge 2)",
    "ZDT6": r"x_1 \in [0, 1],\ x_i = 0\ (i \ge 2)",
}

# keyword arguments needed to construct classes that have no default
# constructor, and how to describe their (variable) shape in the tables
SPECIAL: dict[str, dict[str, Any]] = {
    "Gaussian": {
        "kwargs": {"centers": np.array([[1.0, 0.0], [-1.0, 0.0]])},
        "nobj": "K",
        "dim": "N",
    },
}

TEXT = {
    "en": {
        "class": "Class",
        "aliases": "Aliases",
        "nobj": "Objectives",
        "dim": "Variables",
        "constraints": "Constraints",
        "sense": "Original sense",
        "domain": "Default search space",
        "pareto": "Pareto-optimal set",
        "minimum_point": "Global minimum point",
        "minimum_value": "Minimum value",
        "yes": "yes",
        "no": "no",
        "minimize": "minimization",
        "maximize": "maximization",
        "variable_dim": "N (default {d})",
        "alias_domain": "search space {domain}",
        "gaussian_shape": "given by ``centers`` (K objectives, N variables)",
    },
    "ja": {
        "class": "クラス",
        "aliases": "別名",
        "nobj": "目的数",
        "dim": "変数の数",
        "constraints": "制約",
        "sense": "原著の向き",
        "domain": "既定の探索範囲",
        "pareto": "Pareto 最適集合",
        "minimum_point": "大域最小点",
        "minimum_value": "最小値",
        "yes": "あり",
        "no": "なし",
        "minimize": "最小化",
        "maximize": "最大化",
        "variable_dim": "N（既定 {d}）",
        "alias_domain": "探索範囲 {domain}",
        "gaussian_shape": "``centers`` で指定（目的数 K、変数 N）",
    },
}


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _classes(module, root):
    """Classes defined in ``module`` deriving from ``root``, in definition order."""
    return [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and obj.__module__ == module.__name__
        and issubclass(obj, root)
        and obj is not root
    ]


def _construct(cls):
    kwargs = SPECIAL.get(cls.__name__, {}).get("kwargs", {})
    return cls(**kwargs)


def _has_variable_dim(cls) -> bool:
    return "dim" in inspect.signature(cls.__init__).parameters


def _has_constraint(cls) -> bool:
    return cls.constraint is not base.TestFunction.constraint


def collect(module, root):
    """Return ``[(primary_class, [alias_classes])]`` for the module.

    A class whose direct base is another class of the module is an alias of it.
    """
    classes = _classes(module, root)
    primaries = [c for c in classes if c.__bases__[0] is root]
    aliases = {p: [] for p in primaries}
    for c in classes:
        if c in primaries:
            continue
        primary = c.__bases__[0]
        while primary not in primaries:
            primary = primary.__bases__[0]
        aliases[primary].append(c)
    return [(p, aliases[p]) for p in primaries]


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _num(v: float) -> str:
    """LaTeX for a number; multiples of pi are written with \\pi."""
    for k in (1, 2):
        if abs(abs(v) - k * np.pi) < 1e-12:
            s = r"\pi" if k == 1 else rf"{k}\pi"
            return "-" + s if v < 0 else s
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


def _interval(lo: float, hi: float) -> str:
    return f"[{_num(lo)}, {_num(hi)}]"


def format_domain(fn, variable_dim: bool) -> str:
    """LaTeX description of the search space of ``fn``."""
    lo, hi = fn.min_X, fn.max_X
    d = fn.dim
    if np.all(lo == lo[0]) and np.all(hi == hi[0]):
        exp = "N" if variable_dim else str(d)
        box = _interval(lo[0], hi[0])
        return box if d == 1 and not variable_dim else f"{box}^{{{exp}}}"
    if d > 2 and np.all(lo[1:] == lo[1]) and np.all(hi[1:] == hi[1]):
        return (
            rf"x_1 \in {_interval(lo[0], hi[0])},\ "
            rf"x_i \in {_interval(lo[1], hi[1])}\ (i \ge 2)"
        )
    return r" \times ".join(_interval(a, b) for a, b in zip(lo, hi))


def _point(x: np.ndarray) -> str:
    return "(" + ", ".join(_num(round(float(v), 4)) for v in x) + ")"


def _ref(cls) -> str:
    return f":class:`~{cls.__module__}.{cls.__name__}`"


def _math(s: str) -> str:
    return f":math:`{s}`" if s else ""


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _list_table(header: list[str], rows: list[list[str]], widths: list[int]) -> str:
    out = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: " + " ".join(str(w) for w in widths),
        "",
    ]
    for row in [header] + rows:
        out.append("   * - " + row[0])
        for cell in row[1:]:
            out.append("     - " + cell)
    out.append("")
    return "\n".join(out)


def multi_objective_table(lang: str) -> str:
    t = TEXT[lang]
    rows = []
    for cls, aliases in collect(multi_objective, multi_objective.MultiTestFunction):
        fn = _construct(cls)
        special = SPECIAL.get(cls.__name__, {})
        variable_dim = _has_variable_dim(cls)

        alias_cells = []
        for a in aliases:
            afn = _construct(a)
            cell = _ref(a)
            if not (np.array_equal(afn.min_X, fn.min_X) and np.array_equal(afn.max_X, fn.max_X)):
                cell += " (" + t["alias_domain"].format(domain=_math(format_domain(afn, variable_dim))) + ")"
            alias_cells.append(cell)

        if "nobj" in special:
            nobj = special["nobj"]
            dim = special["dim"]
            domain = _math(format_domain(fn, True))
        else:
            nobj = str(fn.nobj)
            dim = t["variable_dim"].format(d=fn.dim) if variable_dim else str(fn.dim)
            domain = _math(format_domain(fn, variable_dim))

        rows.append(
            [
                _ref(cls),
                ", ".join(alias_cells),
                nobj,
                dim,
                t["yes"] if _has_constraint(cls) else t["no"],
                t["maximize"] if fn.is_maximization else t["minimize"],
                domain,
                _math(PARETO_SET.get(cls.__name__, "")),
            ]
        )
    header = [t["class"], t["aliases"], t["nobj"], t["dim"], t["constraints"], t["sense"], t["domain"], t["pareto"]]
    return _list_table(header, rows, [18, 16, 6, 9, 7, 9, 17, 22])


def single_objective_table(lang: str) -> str:
    t = TEXT[lang]
    rows = []
    for cls, aliases in collect(single_objective, single_objective.SingleTestFunction):
        fn = _construct(cls)
        variable_dim = _has_variable_dim(cls)
        xopt = np.atleast_2d(fn.global_minimum_point())
        fmin = fn(xopt)
        if fn.test_maximizer:
            fmin = -fmin
        rows.append(
            [
                _ref(cls),
                ", ".join(_ref(a) for a in aliases),
                t["variable_dim"].format(d=fn.dim) if variable_dim else str(fn.dim),
                t["yes"] if _has_constraint(cls) else t["no"],
                _math(format_domain(fn, variable_dim)),
                _math(",\\ ".join(_point(x) for x in xopt)),
                _math(_num(round(float(fmin.min()), 6))),
            ]
        )
    header = [t["class"], t["aliases"], t["dim"], t["constraints"], t["domain"], t["minimum_point"], t["minimum_value"]]
    return _list_table(header, rows, [18, 10, 9, 7, 16, 28, 10])


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

GENERATED_DIR = "_generated"


def write_tables(source_dir: str, lang: str) -> list[str]:
    """Write the tables into ``<source_dir>/_generated`` and return the paths."""
    out_dir = os.path.join(source_dir, GENERATED_DIR)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, table in (
        ("test_functions_multi.rst", multi_objective_table(lang)),
        ("test_functions_single.rst", single_objective_table(lang)),
    ):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(table)
        written.append(path)
    return written


def sphinx_setup(app, lang: str):
    """Hook for ``conf.py``: generate the tables before the sources are read."""

    def _generate(app):
        write_tables(str(app.srcdir), lang)

    app.connect("builder-inited", _generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
