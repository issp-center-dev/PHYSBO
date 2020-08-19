はじめに
=====================


COMBO とは
----------------------

COMBOは、高速でスケーラブルなベイズ最適化 (bayesian optimization) のための Python ライブラリです。

物理や化学分野において、データドリブンな実験計画アルゴリズムによって科学的発見を加速する、という試みが多く行われています。
ベイズ最適化は、このような実験科学における発見を加速するために有効なツールです。

一般的にベイズ最適化は計算コストが高く、scikit-learn 等のスタンダードな実装では、多くのデータを扱うことが困難です。
COMBO は以下の特徴により、高いスケーラビリティを実現しています。

* Thompson Sampling
* random feature map
* one-rank Cholesky update
* automatic hyperparameter tuning

技術的な詳細については、`こちらの文献 <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ を参照して下さい。


COMBO の引用
----------------------

COMBO を利用にあたっては、以下の文献を引用してください、

[ref] Tsuyoshi Ueno, Trevor David Rhone, Zhufeng Hou, Teruyasu Mizoguchi and Koji Tsuda,
COMBO: An Efficient Bayesian Optimization Library for Materials Science,
Materials Discovery, (2016). Available from http://dx.doi.org/10.1016/j.md.2016.04.001

Bibtex は以下の通りです。 ::

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


ライセンス
----------------------
COMBO は `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_ に従って配布されています。

`The MIT License (MIT) <https://opensource.org/licenses/MIT>`_.

Copyright (c) <2015-> <`Tsuda Laboratory <http://www.tsudalab.org/>`_>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

