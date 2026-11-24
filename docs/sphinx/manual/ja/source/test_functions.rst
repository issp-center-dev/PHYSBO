.. _chap_test_functions:

ベンチマーク関数
=====================

PHYSBO には、最適化アルゴリズムを試したり比較したりするためのベンチマーク（テスト）関数が
:mod:`physbo.test_functions` に用意されています。
以下の表は、関数を選ぶ際に必要な性質をまとめたものです。
各関数の定義、参考文献、超体積の計算に用いる参照箱は API リファレンス
(:mod:`physbo.test_functions.multi_objective` および
:mod:`physbo.test_functions.single_objective`) に記載されており、
表のクラス名からリンクされています。

使い方
-----------

テスト関数オブジェクトは形状 ``(n, dim)`` の点の配列に対して呼び出すことができ、
形状 ``(n, nobj)`` の配列を返します。
また、探索範囲 (``min_X``, ``max_X``)、制約によるフィルタ (``constraint``)、
制約を適用した格子点の生成 (``make_grid``)、多目的関数では超体積の参照箱
(``reference_min``, ``reference_max``) を提供します。

.. code-block:: python

   import physbo

   fn = physbo.test_functions.multi_objective.SRN()
   X = fn.make_grid(101)      # 制約を満たす候補点
   Y = fn(X)                  # 形状 (N, 2)
   # ... 探索の後 ...
   vid = res.pareto.volume_in_dominance(fn.reference_min, fn.reference_max)

各関数は、従った文献と同じ向き（最小化または最大化）で実装されています
（表の「原著の向き」の列）。
PHYSBO は目的関数を最大化するので、テスト関数が返す値は既定
(``test_maximizer=True``) では最大化問題として表されます。
最小化問題の関数は符号を反転した値が、最大化問題の関数はそのままの値が返ります。
``test_maximizer=False`` を指定すると、最小化問題としての値が返ります。
参照箱も同じ規約に従います。

多目的関数
-----------

別名の探索範囲は、「別名」の列に注記がない限り、参照先のクラスと同じです。
Pareto 最適集合は、閉じた形で知られているものについて記載しています。

.. include:: _generated/test_functions_multi.rst

単目的関数
-----------

大域最小点と最小値は最小化問題 (``test_maximizer=False``) としての値です。
変数の数を変えられる関数については、既定の変数の数での値を示しています。

.. include:: _generated/test_functions_single.rst
