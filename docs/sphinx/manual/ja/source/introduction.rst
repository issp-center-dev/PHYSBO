はじめに
=====================

PHYSBO とは
----------------------

PHYSBOは、高速でスケーラブルなベイズ最適化 (bayesian optimization) のためのPythonライブラリです。
COMBO(COMmon Baysian Optimization)をもとに、主に物性分野の研究者をターゲットに開発されました。
物理や化学分野において、データドリブンな実験計画アルゴリズムによって科学的発見を加速する、という試みが多く行われています。
ベイズ最適化は、このような実験科学における発見を加速するために有効なツールです。
一般的にベイズ最適化は計算コストが高く、scikit-learn 等のスタンダードな実装では、多くのデータを扱うことが困難です。
PHYSBOでは以下の特徴により、高いスケーラビリティを実現しています。

* Thompson Sampling
* random feature map
* one-rank Cholesky update
* automatic hyperparameter tuning

技術的な詳細については、`こちらの文献 <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ を参照して下さい。


PHYSBO の引用
----------------------

PHYSBOを引用する際には、以下の文献を引用してください、

Tsuyoshi Ueno, Trevor David Rhone, Zhufeng Hou, Teruyasu Mizoguchi and Koji Tsuda,
COMBO: An Efficient Bayesian Optimization Library for Materials Science,
Materials Discovery, (2016). Available from http://dx.doi.org/10.1016/j.md.2016.04.001

Bibtexは以下の通りです。 ::

    @article{Ueno2016,
    title = "COMBO: An Efficient Bayesian Optimization Library for Materials Science ",
    journal = "Materials Discovery",
    volume = "",
    number = "",
    pages = "-",
    year = "2016",
    note = "",
    doi = "http://dx.doi.org/10.1016/j.md.2016.04.001",
    url = "http://www.sciencedirect.com/science/article/pii/S2352924516300035",
    author = "Tsuyoshi Ueno and Trevor David Rhone and Zhufeng Hou and Teruyasu Mizoguchi and Koji Tsuda",
    }

主な開発者
----------------------
- ver. 1.0

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

Copyright (c) <2020-> The University of Tokyo.* *All rights reserved.

本ソフトウェアは2020年度 東京大学物性研究所 ソフトウェア高度化プロジェクトの支援を受け開発されました。
