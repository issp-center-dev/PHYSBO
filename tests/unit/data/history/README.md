History files written by earlier versions of PHYSBO, read by
`tests/unit/test_history.py`.

Each directory holds one file per search module, written by
`tests/unit/_history_data.py` run with the PHYSBO of that version
(`PYTHONPATH=<checkout>/src python tests/unit/_history_data.py <directory>`):

- `v3.2.1`: the release 3.2.1.
- `develop_619d68fe`: the development version before the history classes were
  restructured (#156).

The files of the single-objective classes are `.npz` archives; those of the
multi-objective and unified classes are pickles that contain a
`physbo.search.pareto.Pareto` object. They were written with numpy 2.5.1 and
Python 3.13 (pickle protocol 4).
