インストール
=====================

実行環境・必要なパッケージ
----------------------

* Python >= 3.6
* numpy >=1.10
* scipy >= 0.16
* Cython >= 0.22.1
* mpi4py >= 2.0 (optional)

Python 3系 で動作します(推奨バージョン: 3.6 以上)。

`Anaconda <https://www.anaconda.com/>`_  環境を利用すると、numpy, scipy, Cython がデフォルトでインストールされているため、COMBO をすぐに実行することが可能です。
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
----------------------
#. ソースファイルをダウンロードするか、以下のように github レポジトリをクローンしてください  ::
        
   > git clone https://github.com/tsudalab/combo.git

#. setup.py install を実行します::

   > cd combo
   > python setup.py install

Windows でのインストールについて
----------------------

COMBO は Cython による拡張モジュールを利用しているため、インストールには Cコンパイラが必要となります。
Windows版の Python 本体のコンパイルには Microsoft Visual C++ が使われており、COMBOのインストール時にも同じコンパイラを使用する必要があります。

以下のリンクを参照して Visual C++ Build Tools 2017 をインストールしてください。 
https://www.python.jp/install/windows/install_vstools2017.html

なお Python 3.5 以下を利用している場合は、必要な Visual C++ のバージョンが異なりますので注意してください。

アンインストール
----------------------
#. 以下のコマンドを実行します ::

   > pip uninstall combo


