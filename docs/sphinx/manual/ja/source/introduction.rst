はじめに
=====================

PHYSBO とは
----------------------

PHYSBO(optimization tool for PHYSics based on Bayesian Optimization)は、高速でスケーラブルなベイズ最適化 (Bayesian optimization) のためのPythonライブラリです。
COMBO(COMmon Baysian Optimization)をもとに、主に物性分野の研究者をターゲットに開発されました。
物理、化学、材料分野において、データ駆動的な実験計画アルゴリズムによって科学的発見を加速する、という試みが多く行われています。
ベイズ最適化は、このような科学的発見を加速するために有効なツールです。
ベイズ最適化は、複雑なシミュレーションや、実世界における実験タスクなど、目的関数値（特性値など）の評価に大きなコストがかかるような場合に利用できる手法です。つまり、「できるだけ少ない実験・シミュレーション回数でより良い目的関数値（材料特性など）を持つパラメータ（材料の組成、構造、プロセスやシミュレーションパラメータなど）を見つけ出す」ことが、ベイズ最適化によって解かれる問題です。ベイズ最適化では、探索するパラメータの候補をあらかじめリストアップし、候補の中から目的関数値が最大と考えられる候補を機械学習（ガウス過程回帰を利用）による予測をうまく利用することで選定します。その候補に対して実験・シミュレーションを行い目的関数値を評価します。機械学習による選定・実験シミュレーションによる評価を繰り返すことにより、少ない回数での最適化が可能となります。
一方で、一般的にベイズ最適化は計算コストが高く、scikit-learn 等のスタンダードな実装では、多くのデータを扱うことが困難です。
PHYSBOでは以下の特徴により、高いスケーラビリティを実現しています。

* Thompson Sampling
* random feature map
* one-rank Cholesky update
* automatic hyperparameter tuning

技術的な詳細については、`こちらの文献 <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ を参照して下さい。


PHYSBO の引用
----------------------

PHYSBOを引用する際には、以下の文献を引用してください、


Yuichi Motoyama, Ryo Tamura, Kazuyoshi Yoshimi, Kei Terayama, Tsuyoshi Ueno, Koji Tsuda,
Bayesian optimization package: PHYSBO,
Computer Physics Communications Volume 278, September 2022, 108405. Available from https://www.sciencedirect.com/science/article/pii/S0010465522001242?via%3Dihub (open access).

Bibtexは以下の通りです。 ::

  @misc{@article{MOTOYAMA2022108405,
  title = {Bayesian optimization package: PHYSBO},
  journal = {Computer Physics Communications},
  volume = {278},
  pages = {108405},
  year = {2022},
  issn = {0010-4655},
  doi = {https://doi.org/10.1016/j.cpc.2022.108405},
  author = {Yuichi Motoyama and Ryo Tamura and Kazuyoshi Yoshimi and Kei Terayama and Tsuyoshi Ueno and Koji Tsuda},
  keywords = {Bayesian optimization, Multi-objective optimization, Materials screening, Effective model estimation}
  }
  
主な開発者
----------------------
- ver. 1.0-

  - 田村 亮 (物質・材料研究機構 国際ナノアーキテクトニクス研究拠点)
  - 寺山 慧 (横浜市立大学大学院 生命医科学研究科)
  - 津田 宏治 (東京大学大学院 新領域創成科学研究科)
  - 植野 剛 (株式会社 Magne-Max Capital Management)
  - 本山 裕一 (東京大学 物性研究所)
  - 吉見 一慶 (東京大学 物性研究所)
  - 川島 直輝 (東京大学 物性研究所)


- ver. 0.1-0.3

  - 田村 亮 (物質・材料研究機構 国際ナノアーキテクトニクス研究拠点)
  - 寺山 慧 (横浜市立大学大学院 生命医科学研究科)
  - 津田 宏治 (東京大学大学院 新領域創成科学研究科)
  - 本山 裕一 (東京大学 物性研究所)
  - 吉見 一慶 (東京大学 物性研究所)
  - 川島 直輝 (東京大学 物性研究所)

ライセンス
----------------------
| 本ソフトウェアのプログラムパッケージおよびソースコード一式はGNU
  General Public License version 3（GPL v3）に準じて配布されています。

Copyright (c) <2020-> The University of Tokyo. All rights reserved.

本ソフトウェアは2020年度 東京大学物性研究所 ソフトウェア高度化プロジェクトの支援を受け開発されました。
