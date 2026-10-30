.. _chap_random_numbers:

乱数と再現性
==================================

探索のいくつかの段階で乱数を使います。``random_search`` での候補の選択、
ハイパーパラメータ学習の初期値とミニバッチ、ベイズ線形モデル
(``num_rand_basis > 0``)のランダムフーリエ基底、Thompson sampling
(``score="TS"``)の事後サンプル、1 ステップに複数点を提案するとき
(``num_search_each_probe > 1``)の仮想観測値、range 系 policy における
獲得関数のランダム最適化、``ParEGO`` の重みです。

これらの乱数はすべて、policy がコンストラクタの ``rng`` 引数で選ぶ
乱数生成器から引かれます。``rng`` 引数はすべての policy
(``discrete``、``discrete_multi``、``discrete_unified``、``range``、
``range_multi``、``range_unified``)が受け付けます。

2 つのモード
----------------------------------

**legacy モード(デフォルト)。**
``rng`` を与えない(または ``rng="legacy"``)場合、policy は従来の
PHYSBO と同じくグローバルな ``numpy.random`` の状態から乱数を引きます。
``set_seed()`` はこのグローバル状態に seed を設定するので、
``numpy.random.seed`` や ``policy.set_seed`` を呼ぶスクリプトは
これまでと同じ結果を与え続けます。この状態は、プロセス内で
``numpy.random`` を使う他のすべてのコードと共有されます。

**Generator モード。**
``rng`` に整数の seed、``numpy.random.SeedSequence``、または
``numpy.random.Generator`` を与えると、policy は自分専用の
``numpy.random.Generator`` を持ち、乱数をそこからだけ引きます。
グローバルな ``numpy.random`` の状態は読みも書きもしないので、
プロセスの他の部分が ``numpy.random`` をどう使っても探索は再現され、
生成器は policy オブジェクトの一部になります。policy を pickle すると
その状態も保存されます(:ref:`chap_checkpoint` を参照)。
このモードでは ``set_seed(seed)`` は seed から生成器を作り直します。
乱数列は legacy モードとは異なるので、Generator モードでの実行は
同じ seed を使っても以前のバージョンの実行を再現しません。

.. code-block:: python

    # legacy モード: グローバルな numpy.random の状態を set_seed() で設定
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(0)

    # Generator モード: seed 12345 で初期化した policy 専用の生成器
    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)

    # Generator モード: 呼び出し側で作った生成器を渡す
    rng = numpy.random.default_rng(12345)
    policy = physbo.search.discrete.Policy(test_X=X, rng=rng)

どちらのモードでも、同じ PHYSBO のバージョンを同じ環境
(Python、NumPy、SciPy のバージョン、BLAS/LAPACK ライブラリ)で使えば、
同じ seed は同じ探索を与えます。

補助オブジェクト
----------------------------------

policy に渡すオブジェクトのうち自身で乱数を引くものは、独自の生成器を
与えられていない限り policy の生成器に従います。

- ``physbo.search.unify.ParEGO`` は ``rng`` 属性を通して重みを引きます。
  ``rng`` 引数なしで ``bayes_search`` の ``unify_method`` に渡すと
  policy の生成器を使い、``ParEGO(..., rng=...)`` とすれば与えた生成器を
  使い続けます。
- range 系 policy のデフォルトのランダム最適化器は policy の生成器で
  作られます。呼び出し側で作る最適化器は自身の ``rng`` 引数を取ります。
  例: ``physbo.search.optimize.random.Optimizer(min_X, max_X, nsamples=1000, rng=policy.rng)``
- 組み込みの Thompson sampling オブジェクト
  (``physbo.search.score.ThompsonSampling`` と
  ``physbo.search.score_multi.ThompsonSampling``)も同様に ``rng`` 属性を
  通して乱数を引きます。名前 ``"TS"`` と ``rng`` なしで作ったインスタンスは
  policy の生成器を使い、``ThompsonSampling(..., rng=...)`` とすれば与えた
  生成器を使い続けます。ユーザー定義のスコア(アルゴリズムの章を参照)は
  自身が使う乱数源から引き、policy の生成器とは結び付きません。

並列実行
----------------------------------

並列実行(``comm`` にコミュニケータを渡した場合)では、すべての rank で
policy に同じ ``rng`` 指定を与えてください。全 rank が一致させる必要の
ある量は rank 0 で引いてブロードキャストします。モデルのハイパー
パラメータ、ランダムフーリエ基底、ベイズ線形モデルでの Thompson
sampling の事後サンプル、離散 policy における複数点提案の仮想観測値、
ランダム最適化器が返す点、``constraint`` つきの range 系 policy における
``random_search`` のランダムな点(他の rank は同じ個数の点を引き、生成器を
rank 0 と揃えます)、シミュレータの結果がこれにあたります。それ以外の乱数は各 rank
が自分の担当する候補について引きます。その結果、各 rank の乱数列は
互いに揃いません(rank 0 が他より多く引きます)が、複数の rank の探索に
影響する量はすべてブロードキャストされるので問題ありません。

したがってベイズ線形モデル(``num_rand_basis > 0``)では、Thompson
sampling は rank 数によらず、同じ seed の逐次実行と同じ点を提案します。
ガウス過程の predictor(``num_rand_basis == 0``)では事後サンプルに
ブロードキャストできる有限次元の表現がないため、各 rank は自分の候補に
ついてだけ事後分布からサンプルします。異なる rank 上の候補間の相関は
無視され、提案は rank 数に依存します。並列実行で Thompson sampling を
使う場合はベイズ線形モデルを推奨します。

``get_score`` は ``parallel=True``(デフォルト)では候補を rank に分割し、
Thompson sampling のサンプルを rank 0 で引きます。``parallel=False`` では
各 rank が与えられた候補を自分のサンプルで採点します。
