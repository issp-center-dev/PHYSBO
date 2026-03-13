学習の詳細設定
======================

PHYSBOではガウス過程のハイパーパラメータ学習 ("learning") を行いますが、学習手法自体にも例えばadamの学習率など、ハイパーパラメータが存在します。
これらのハイパーパラメータを変更する場合には ``physbo.SetConfig`` を使用します。
なお、基本的には既定値から変更する必要はありません。

``SetConfig`` の使い方
----------------------

まずは ``config = physbo.SetConfig()`` としてインスタンスを作成します。すべてのパラメータに既定値が設定されているので、変更したいパラメータを直接変更します。
作成した ``config`` を、 ``Policy`` のコンストラクタの ``config`` 引数に渡します。
例えばadamの学習率を0.01に変更する場合は、以下のようになります。

.. code-block:: python

   config = physbo.SetConfig()
   config.adam.alpha = 0.01
   policy = physbo.search.discrete.Policy(test_X, config=config)

パラメータをINIファイルに保存し、 ``config.load("config.ini")`` で読み込むこともできます。
例えば以下のようなINIファイルを作成します。

.. code-block:: ini

   [adam]
   alpha = 0.01


設定可能パラメータ
------------------

INIファイルはセクションとキーの階層構造になっています。
これに対応して、 ``SetConfig`` も階層構造を持っています。
例えば ``[adam]`` セクションの ``alpha`` キーは、 ``config.adam.alpha`` メンバ変数に対応します。

セクションやパラメータについて詳しくは以下のとおりです。

``[learning]`` -- 学習の共通設定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ガウス過程回帰モデルの学習に関する共通設定です。
現在、学習手法 ``method`` は adam, bfgs, batch の3種類をサポートしています。
adamはオンライン学習で、bfgsとbatchはバッチ学習です。

.. csv-table::
   :header: キー, 型, 既定値, 説明
   :widths: 26, 8, 12, 60

   ``method``, str, "adam", "学習手法。 ``adam``, ``bfgs``, ``batch`` のいずれか"
   ``is_disp``, bool, True, "学習時の表示の有無。 ``true``, ``false`` のいずれか"
   ``num_disp``, int, 10, 表示間隔（何反復ごとに表示するか）
   ``num_init_params_search``, int, 20, 初期ハイパーパラメータ探索の反復数

``[online]`` -- オンライン学習の共通設定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

オンライン学習の共通設定です。
``method=adam`` のときに使われます。

.. csv-table::
   :header: キー, 型, 既定値, 説明
   :widths: 32, 8, 10, 42

   ``max_epoch``, int, 500, 最大エポック数
   ``max_epoch_init_params_search``, int, 50, 初期ハイパーパラメータ探索の最大エポック数
   ``batch_size``, int, 64, バッチサイズ
   ``eval_size``, int, 5000, 評価用サンプル数

``[adam]`` -- adam 学習
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

adamの設定です。
``method=adam`` のときに使われます。

.. csv-table::
   :header: キー, 型, 既定値, 説明
   :widths: 12, 8, 10, 42

   ``alpha``, float, 0.001, 学習率
   ``beta``, float, 0.9, 一次モーメントの減衰率
   ``gamma``, float, 0.999, 二次モーメントの減衰率
   ``epsilon``, float, 1e-6, 数値計算を安定化させるための微小量

``[batch]`` -- batch / BFGS 学習
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bfgsとbatchの設定です。
``method=bfgs`` または ``method=batch`` のときに使われます。

.. csv-table::
   :header: キー, 型, 既定値, 説明
   :widths: 32, 8, 10, 42

   ``max_iter``, int, 200, 最大反復数
   ``max_iter_init_params_search``, int, 20, 初期ハイパーパラメータ探索の最大反復数
   ``batch_size``, int, 5000, バッチサイズ


``[search]`` -- ベイズ最適化でつかうパラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: キー, 型, 既定値, 説明
   :widths: 28, 8, 10, 60

   ``multi_probe_num_sampling``, int, 20, ``num_search_each_probe>1`` のときに使われる。未評価の目的関数の値を計算するためのサンプル数
   ``alpha``, float, 1.0, Thompson Samplingのハイパーパラメータ。 サンプリング時に事後分布の標準偏差をリスケールする係数
