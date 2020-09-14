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

`Anaconda <https://www.anaconda.com/>`_  環境を利用すると、numpy, scipy, Cython がデフォルトでインストールされているため、PHYSBOをすぐに実行することが可能です。
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
        
   > git clone https://github.com/issp-center-dev/PHYSBO.git

#. setup.py install を実行します::

   > cd physbo
   > python setup.py install

アンインストール
----------------------
#. 以下のコマンドを実行します ::

   > pip uninstall physbo


