基本的な使用方法
=====================

インストール
---------------------

実行環境・必要なパッケージ
~~~~~~~~~~~~~~~~~~~~~~~~

* Python = 2.7
* numpy >=1.10
* scipy >= 0.16
* Cython >= 0.22.1
* mpi4py >= 2.0 (optional)

Python 2.7 で動作します。

.. `Anaconda <https://www.anaconda.com/>`_  環境を利用すると、numpy, scipy, Cython がデフォルトでインストールされているため、COMBO をすぐに実行することが可能です。
   依存パッケージを手動でインストールする場合は、以下の手順によりまとめてインストールすることができます。

   #. 以下をコピーして、'requirements.txt' というファイル名で保存します (setup.py と同じディレクトリ内に保存します） ::

        ## To install these requirements, run
        ## pip install -U -r requirements.txt
        ## (the -U option also upgrades packages; from the second time on,
        ## just run
        ## pip install -r requirements.txt
        ##
        ## NOTE: before running the command above, you need to install a recent version
        ## of pip from the website, and then possibly install/upgrade setuptools using
        ## sudo pip install --upgrade setuptools
        ## numpy
        numpy >=1.10
        
        ## scipy
        scipy >= 0.16
        
        ##  
        Cython >= 0.22.1
        
        ## mpi4py 
        mpi4py >= 2.0 (optional)

   #. 以下のコマンドを実行します。 :: 

    > pip install -U -r requirements.txt

インストール
~~~~~~~~~~~~~~~~~~~~~~~~

#. ソースファイルをダウンロードするか、以下のように github レポジトリをクローンしてください  ::
        
   > git clone https://github.com/issp-center-dev/PHYSBO.git

#. setup.py install を実行します::

   > cd physbo
   > python setup.py install

アンインストール
~~~~~~~~~~~~~~~~~~~~~~~~

#. 以下のコマンドを実行します ::

   > pip uninstall physbo


PHYSBOの基本構造
--------------------------

PHYSBOは以下のような構成になっています(第2階層まで表示)。

..
 |--physbo
 |    |--blm
 |    |--gp
 |    |--misc
 |    |--opt
 |    |--search
 |    |--predictor.py
 |    |--variable.py

各モジュールは以下のような構成で作成されています。
 
- ``blm`` :Baysean linear modelに関するモジュール
- ``gp`` :Gaussian Processに関するモジュール
- ``opt`` :最適化に関するモジュール
- ``search`` :最適解を探索するためのモジュール
- ``predictor.py`` :predictorの抽象クラス、
- ``variable.py`` :physboで用いる変数関連について定義されたクラス
- ``misc`` : その他(探索空間を正規化するためのモジュールなど)
 
各モジュールの詳細についてはAPIリファレンスを参考にしてください。
 
計算の流れ
--------------------------

ベイズ最適化は、複雑なシミュレーションや、実世界における実験タスクなど、目的関数の評価に大きなコストがかかるような最適化問題に適しています。

PHYSBO では以下の手順により最適化を実行します(それぞれの詳細はチュートリアルおよびAPIリファレンスを参考にしてください)。

1. 探索空間の定義

  N: 探索候補の数 , d: 入力パラメータの次元数 とした時、探索候補である各パラメータセット (d 次元のベクトル) を定義します。

2. simulator の定義

  上で定義した探索候補に対して、各探索候補の負のエネルギー(最適化したい目的関数の値)を与えるsimulatorを定義します。
  PHYSBO では最適化の方向は「最大化」だと仮定します。元々の問題設定は「エネルギー最小化」ですが、PHYSBOで最適化を行うにあたって、目的関数値にマイナスを掛けて「負のエネルギーの最大化」問題として扱っています。

3. 最適化の実行

  最初に、最適化の policy をセットします(探索空間はこの段階で引数としてpolicyに渡されます)。最適化方法は、以下の2種類から選択します。
  
  - `random_search`  
  - `bayes_search`
  
  `random_search` では、探索空間からランダムにパラメータを選び、その中で最小となるスコアを探します。主に、ベイズ最適化を行うための前処理として初期パラメータ群をゲットするために使用します。`bayes_search` は、ベイズ最適化を行います。ベイズ最適化でのscore: 獲得関数(acquisition function) のタイプは、以下のいずれかから指定します。

  - TS (Thompson Sampling)
  - EI (Expected Improvement)
  - PI (Probability of Improvement)
  
  各手法の詳細については、`こちらの文献 <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_  およびその参考文献を参照して下さい。

  これらのメソッドに先ほど定義した simulator と探索ステップ数を指定すると、探索ステップ数だけ以下のループが回ります。

    i). パラメータ候補の中から次に実行するパラメータを選択
    
    ii). 選択されたパラメータで simulator を実行

  i)で返されるパラメータはデフォルトでは1つですが、1ステップで複数のパラメータを返すことも可能です。詳しくは「複数候補を一度に探索する」の項目を参照してください。また、上記のループを PHYSBO の中で回すのではなく、i) と ii) を別個に外部から制御することも可能です。つまり、PHYSBO から次に実行するパラメータを推薦してもらい、それをPHYSBOの外部で何らかの形で評価し、評価値をPHYSBOに登録する、という手順が可能です。詳しくは、チュートリアルの「インタラクティブに実行する」の項目を参照してください。
  
    
4. 結果の確認

  探索結果 res は history クラスのオブジェクト (physbo.search.discrete.results.history) として返されます。以下より探索結果を参照します。

  - res.fx : simulator (目的関数) の評価値の履歴
  - res.chosed_actions: simulator を評価したときのパラメータの履歴
  - fbest, best_action= res.export_all_sequence_best_fx(): simulator を評価した全タイミングにおけるベスト値とそのパラメータの履歴
  - res.total_num_search: simulator のトータル評価数

  また、探索結果は save メソッドにより外部ファイルに保存できます。load メソッドを用いて結果ファイルをロードすることができます。使用方法の詳細はチュートリアルをご覧ください。 


