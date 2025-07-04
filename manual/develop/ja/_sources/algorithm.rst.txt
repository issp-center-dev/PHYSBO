.. _chap_algorithm:

アルゴリズム
=====================
ここでは、ベイズ最適化に関する説明を行います。技術的な詳細については、`こちらの文献 <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ を参照してください。

ベイズ最適化
---------------------
ベイズ最適化は、複雑なシミュレーションや、実世界における実験タスクなど、目的関数（特性値など）の評価に大きなコストがかかるような場合に利用できる手法です。つまり、「できるだけ少ない実験・シミュレーション回数でより良い目的関数（材料特性など）を持つ説明変数（材料の組成、構造、プロセスやシミュレーションパラメータなど）を見つけ出す」ことが、ベイズ最適化によって解かれる問題です。ベイズ最適化では、探索する説明変数（ベクトル :math:`{\bf x}` で表す）の候補をあらかじめリストアップした状況からスタートします。そして、候補の中から目的関数 :math:`y` が良くなると考えられる候補を、機械学習（ガウス過程回帰を利用）による予測をうまく利用することで選定します。その候補に対して実験・シミュレーションを行い目的関数の値を評価します。機械学習による選定・実験シミュレーションによる評価を繰り返すことにより、できるだけ少ない回数で最適化が可能となります。


ベイズ最適化のアルゴリズムの詳細を以下に示します。
- ステップ１：初期化

探索したい空間をあらかじめ用意します。つまり、候補となる材料の組成・構造・プロセスやシミュレーションパラメータ等を、ベクトル :math:`{\bf x}` で表現しリストアップします。この段階では、目的関数の値はわかっていません。このうち初期状態としていくつかの候補を選び、実験またはシミュレーションによって目的関数の値 :math:`y` を見積もります。これにより、説明変数 :math:`{\bf x}` と目的関数 :math:`y` が揃った学習データ :math:`D = \{ {\bf x}_i, y_i \}_{(i=1, \cdots, N)}` が得られます。

- ステップ２：候補選定

学習データを用いて、ガウス過程を学習します。ガウス過程によれば、任意の :math:`{\bf x}` における予測値の平均を :math:`\mu_c ({\bf x})` 、分散を :math:`\sigma_c ({\bf x})` とすると、

.. math::
   
   \mu_c ({\bf x}) &= {\bf k}({\bf x})^T (K+\sigma^2 I)^{-1}{\bf y}

   \sigma_c({\bf x}) &= k({\bf x}, {\bf x}) + \sigma^2 - {\bf k}({\bf x})^T  (K+\sigma^2 I)^{-1}{\bf k}({\bf x})

となります。ただし、 :math:`k({\bf x}, {\bf x}')` はカーネルと呼ばれる関数であり、２つのベクトルの類似度を表します。一般に、以下のガウスカーネルが使われます。

.. math::

   k({\bf x}, {\bf x}') = \exp \left[ -\frac{1}{2\eta^2}||{\bf x} - {\bf x}'||^2 \right]

また、このカーネル関数を利用し、 :math:`{\bf k}({\bf x})` および :math:`K` は以下のように計算されます。

.. math::
   
   {\bf k}({\bf x}) = \left( k({\bf x}_1, {\bf x}), k({\bf x}_2, {\bf x}), \cdots, k({\bf x}_N, {\bf x}) \right)^\top

.. math::
   :nowrap:

    \[
    K = \left(
    \begin{array}{cccc}
       k({\bf x}_1, {\bf x}_1) & k({\bf x}_1, {\bf x}_2) & \ldots &  k({\bf x}_1, {\bf x}_N) \\
       k({\bf x}_2, {\bf x}_1) & k({\bf x}_2, {\bf x}_2) & \ldots &  k({\bf x}_2, {\bf x}_N) \\
      \vdots & \vdots & \ddots & \vdots \\
       k({\bf x}_N, {\bf x}_1) & k({\bf x}_N, {\bf x}_2) & \ldots &  k({\bf x}_N, {\bf x}_N)
    \end{array}
    \right)
    \]

まだ実験やシミュレーションを行っていない候補全てに対して、予測値 :math:`\mu_c ({\bf x})` および予測の不確かさに関連する分散 :math:`\sigma_c ({\bf x})` を見積もります。これを用いて獲得関数を計算し、目的関数の値がまだわかっていない候補の中から、獲得関数を最大化する候補 :math:`{\bf x}^*` を選定します。このとき、 :math:`\sigma` および :math:`\eta` はハイパーパラメタと呼ばれ、PHYSBOでは最適な値が自動で設定されます。

獲得関数として、例えば、最大改善確率(PI : Probability of Improvement)、最大期待改善率(EI : Expected Improvement)が有用です。
PIによるスコアは次のように定義される。

.. math::

   \text{PI} (\mathbf{x}) = \Phi (z (\mathbf{x})), \ \ \ z(\mathbf{x}) = \frac{\mu_c (\mathbf{x}) - y_{\max}}{\sigma_c (\mathbf{x})}
   

ここで :math:`\Phi(\cdot)` は累積分布関数です。
PIスコアは、現在得られている :math:`y` の最大値 :math:`y_{\max}` を超える確率を表します。
さらに、EIによるスコアは、予測値と現在の最大値 :math:`y_{\max}` との差の期待値であり、次式で与えられます。

.. math::

   \text{EI} (\mathbf{x}) = [\mu_c (\mathbf{x})-y_{\max}] \Phi (z (\mathbf{x})) + \sigma_c (\mathbf{x}) \phi (z (\mathbf{x})), \ \ \ z(\mathbf{x}) = \frac{\mu_c (\mathbf{x}) - y_{\max}}{\sigma_c (\mathbf{x})}

ここで :math:`\phi(\cdot)` は確率密度関数です。


- ステップ３：実験

ステップ２で選定された獲得関数が最大となる候補 :math:`{\bf x}^*` に対して実験またはシミュレーションを行い、目的関数値 :math:`y` を見積もります。これにより学習データが一つ追加されます。このステップ２、３を繰り返すことで、スコアのよい候補を探索します。


PHYSBOによるベイズ最適化の高速化
---------------------------------------

PHYSBOでは、random feature map、トンプソンサンプリング、コレスキー分解を利用することで、
ベイズ最適化の高速化を実現しています。
まず、random feature mapについて説明します。
random feature map :math:`\phi (\mathbf{x})` を導入することで
ガウスカーネル :math:`k(\mathbf{x},\mathbf{x}')` を以下のように近似しています。

.. math::

   k(\mathbf{x},\mathbf{x}') = \exp \left[ - \frac{1}{2 \eta^2} \| \mathbf{x} -\mathbf{x}' \| \right]^2  \simeq \phi (\mathbf{x})^\top \phi(\mathbf{x}') \\
   \phi (\mathbf{x}) = \left( z_{\omega_1, b_1} (\mathbf{x}/\eta),..., z_{\omega_l, b_l} (\mathbf{x}/\eta) \right)^\top

ただし、 :math:`z_{\omega, b} (\mathbf{x}) = \sqrt{2} \cos (\boldsymbol{\omega}^\top \mathbf{x}+b)` としています。
このとき、 :math:`\boldsymbol{\omega}` は :math:`p(\boldsymbol{\omega}) = (2\pi)^{-d/2} \exp (-\|\boldsymbol{\omega}\|^2/2)` より生成され、 :math:`b` は :math:`[0, 2 \pi]` から一様に選ばれます。
この近似は、 :math:`l \to \infty` の極限で厳密に成立し、 :math:`l` の値がrandom feature mapの次元となります。

このとき :math:`\Phi` を、以下のように学習データのベクトル :math:`\mathbf{x}` による :math:`\phi(\mathbf{x}_i)` を各列に持つ :math:`l` 行 :math:`n` 列行列とします。

.. math::

   \Phi = ( \phi(\mathbf{x}_1),..., \phi(\mathbf{x}_n) )

すると、

.. math::

   \mathbf{k} (\mathbf{x}) = \Phi^\top \phi(\mathbf{x}) \\
   K= \Phi^\top \Phi

という関係が成立することがわかります。

次に、トンプソンサンプリングを利用することで、候補の予測にかかる計算時間を :math:`O(l)` にする手法について紹介します。
EIやPIを利用すると、分散を評価する必要があるため :math:`O(l^2)` になってしまうことに注意が必要です。
トンプソンサンプリングを行うために、
以下で定義されるベイズ線形モデルを利用します。

.. math::

   y = \mathbf{w}^\top \phi (\mathbf{x})

ただし、この :math:`\phi(\mathbf{x})` は前述したrandom feature mapであり、 :math:`\mathbf{w}` は係数ベクトルです。
ガウス過程では、学習データ :math:`D` があたえられたとき、この :math:`\mathbf{w}` が以下のガウス分布に従うように決まります。

.. math::

   p(\mathbf{w}|D) = \mathcal{N} (\boldsymbol{\mu}, \Sigma) \\
   \boldsymbol{\mu} = (\Phi \Phi^\top + \sigma^2 I)^{-1} \Phi \mathbf{y} \\
   \Sigma = \sigma^2 (\Phi \Phi^\top + \sigma^2 I)^{-1}

トンプソンサンプリングでは、この事後確率分布にしたがって係数ベクトルを一つサンプリングし、
それを :math:`\mathbf{w}^*` とすることで、
獲得関数を

.. math::

   \text{TS} (\mathbf{x}) = {\mathbf{w}^*}^\top \phi (\mathbf{x})

と表す。
これを最大とする :math:`\mathbf{x}^*` が次の候補として選出されます。
このとき、 :math:`\phi (\mathbf{x})` は :math:`l` 次元ベクトルなため、
獲得関数の計算は :math:`O(l)` で実行できます。

次に :math:`\mathbf{w}` のサンプリングの高速化について紹介します。
行列 :math:`A` を以下のように定義します。

.. math::

   A = \frac{1}{\sigma^2} \Phi \Phi^\top +I

すると、事後確率分布は、

.. math::

   p(\mathbf{w}|D) = \mathcal{N} \left( \frac{1}{\sigma^2} A^{-1} \Phi \mathbf{y}, A^{-1} \right)

と表すことができます。
そのため、 :math:`\mathbf{w}` をサンプリングするためには、 :math:`A^{-1}` の計算が必要となります。
ここで、ベイズ最適化のイテレーションにおいて、
新しく :math:`(\mathbf{x}', y')` が加わった場合について考えます。
このデータの追加により、行列 :math:`A` は、

.. math::

   A' = A + \frac{1}{\sigma^2} \phi (\mathbf{x}') \phi (\mathbf{x}')^\top

と更新されます。
この更新は、コレスキー分解( :math:`A= L^\top L` )を用いることで、 :math:`A^{-1}` の計算にかかる時間を :math:`O(l^2)` にすることができます。
もし、 :math:`A^{-1}` をイテレーションごとに毎回計算すると :math:`O(l^3)` の計算が必要になります。
実際、 :math:`\mathbf{w}` をサンプリングする際は、

.. math::

   \mathbf{w}^* = \boldsymbol{\mu} + \mathbf{w}_0

とし、 :math:`\mathbf{w}_0` を :math:`\mathcal{N} (0,A^{-1})` からサンプリングします。
また、 :math:`\boldsymbol{\mu}` は、

.. math::

   L^\top L \boldsymbol{\mu} = \frac{1}{\sigma^2} \Phi \mathbf{y}

を解くことで得られます。
これらの技術を利用することで、学習データ数に対してほぼ線形の計算時間を実現しています。


回帰モデルにおける説明変数の重要度
---------------------------------------

回帰モデル :math:`f(\mathbf{x})` における説明変数（特徴量）の重要度は、それぞれの説明変数をランダムに並べ替えたテストデータに対して、
モデルの予測精度がどれだけ低下するかを評価することで計算できます。

説明変数の数（探索空間の次元）を :math:`D` とし、テストデータの数を :math:`N` とします。
この時、テストデータの入力は :math:`N \times D` の行列 :math:`\mathbf{X}` で表されます。
出力（目的関数）は :math:`N` 次元ベクトル :math:`\mathbf{y}` で表されます。
モデルの予測精度は、モデルの出力とテストデータの出力の平均二乗誤差（MSE）とします。

まず、モデルの予測精度のベースラインとして、テストデータそのものの予測値の平均二乗誤差（MSE）を計算します。

.. math::

   \text{MSE}^{\text{base}} = \frac{1}{N}  \| \mathbf{y} - f(\mathbf{X}) \|^2

:math:`a` 番目の説明変数の重要度を調べるために、入力データのうち :math:`a` 番目の変数のみ、:math:`N` 要素の置換 :math:`P` で並べ替えたデータ :math:`X^{P}` を作成します。

.. math::

   X^{P}_{i,a} &= X_{P(i),a}\\
   X^{P}_{i,b} &= X_{i,b} \ \ \ \text{for} \ b \neq a

:math:`\mathbf{X}^{P}` のテスト誤差は

.. math::

   \text{MSE}^{P}_{a} = \frac{1}{N}  \| \mathbf{y} - f(\mathbf{X}^{P}) \|^2

となります。
Permutation Importance (PI) は、置換 :math:`P` をランダムに :math:`N_\text{perm}` 個生成して、

.. math::

   \text{PI}_{a} = \frac{1}{N_\text{perm}} \sum_{P} \text{MSE}^{P}_{a} - \text{MSE}^{\text{base}}

と計算されます。 :math:`\text{PI}_{a}` が大きければ大きいほど、説明変数 :math:`a` のランダム置換によってモデルの予測精度が悪化していることになり、つまり :math:`a` が重要であることを示します。
