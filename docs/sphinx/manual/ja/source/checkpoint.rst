チェックポイントと再開
==================================

PHYSBO には探索を永続化する方法が 2 つあります。

- ``policy.save()`` / ``policy.load()`` は探索の *結果*
  (history、訓練データ、predictor)を可搬なファイルに保存します。
  後から結果を解析する場合や、新しい探索のウォームスタートに適して
  おり、保存時と異なる MPI プロセス数の実行からも読み込めます。
  ただし乱数生成器の状態は保存されないため、再開した実行は中断しな
  かった場合の実行を再現しません。
- 本節で説明するチェックポイント API は、乱数生成器を含む *実行状態
  の全体* を保存します。中断した探索を **bit-exact に再開**でき、
  再開後の探索は中断しなかった場合と完全に同じ候補を選択します。

乱数生成器のモード
----------------------------------

チェックポイント機構は policy の ``rng`` 引数と連動します。

.. code-block:: python

    # legacy モード(デフォルト): グローバルな numpy.random 状態を使用。
    # set_seed() は従来どおりグローバル状態に seed を設定します。
    policy = physbo.search.discrete.Policy(test_X=X)

    # Generator モード: policy が numpy.random.Generator を保持。
    # 乱数状態は policy 自身に格納されます。
    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)

Generator モードでは乱数状態が policy オブジェクトの一部となるため、
policy を pickle するだけで厳密な再開に必要な情報がすべて保存されま
す。legacy モードでは乱数状態はグローバルな ``numpy.random`` モジュ
ールにあるため、チェックポイント API が明示的に捕捉・復元します
(このため legacy モードのチェックポイントを読み込むと、副作用として
*グローバルな numpy.random 状態が書き換わる* ことに注意してください)。

チェックポイントの保存と復元
----------------------------------

.. code-block:: python

    import physbo

    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)
    policy.random_search(max_num_probes=10, simulator=simulator)
    policy.bayes_search(max_num_probes=20, simulator=simulator, score="TS",
                        num_rand_basis=500)

    # 実行状態の全体を単一ファイルに保存
    policy.save_checkpoint("search.ckpt")

    # ... 後で(別プロセスでも可)...

    policy = physbo.search.discrete.Policy.load_checkpoint("search.ckpt")
    # 中断しなかった場合と完全に同じように継続します
    policy.bayes_search(max_num_probes=20, simulator=simulator, score="TS",
                        num_rand_basis=500)

``load_checkpoint`` は保存に使った policy クラスのクラスメソッドです。
異なる policy クラスで読み込もうとするとエラーになります。チェック
ポイントファイルには PHYSBO のバージョンが記録され、不一致の場合は
警告が出ます(bit-exact な再開が保証されるのは同一バージョン内のみ
です)。チェックポイント形式のバージョンも検証されます。

MPI での利用
----------------------------------

``save_checkpoint`` と ``load_checkpoint`` は *集団操作* であり、
全 rank が呼び出す必要があります。rank ローカルな状態(各 rank の
残り候補と乱数状態)が集約され、rank 0 が単一ファイルを書き込みます。
読み込み時は rank 0 がファイルを読んで全 rank に配布します。

.. code-block:: python

    policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=12345)
    policy.random_search(max_num_probes=10, simulator=simulator)
    policy.save_checkpoint("search.ckpt")     # 全 rank が呼ぶ

    # ... 保存時と同じ MPI プロセス数で再開 ...

    policy = physbo.search.discrete.Policy.load_checkpoint(
        "search.ckpt", comm=comm)             # 全 rank が呼ぶ

再開時の MPI プロセス数は保存時と同じでなければなりません。異なる
場合は ``load_checkpoint`` がエラーを送出します。

.. note::

   BLM predictor(``num_rand_basis > 0``)の場合、Thompson sampling は
   事後分布からのサンプルを rank 0 で生成して全 rank に配布するため、
   探索結果は rank 数に依存しません。GP predictor
   (``num_rand_basis == 0``)の場合、MPI での Thompson sampling は
   rank ローカルな近似であり結果が rank 数に依存します。MPI で TS を
   使う場合は BLM predictor を推奨します。

他のアプリケーションへの組み込み
----------------------------------------

PHYSBO を組み込むアプリケーション(ODAT-SE など)が独自のチェック
ポイント機構を持つ場合は、自身の状態の一部として policy をそのまま
pickle できます。MPI コミュニケータは pickle 時に自動的に除外される
ため、復元後に ``set_comm()`` で再アタッチします。

.. code-block:: python

    # 保存(ホストアプリケーション自身のチェックポイント処理内)
    state = {
        "step": step,
        "policy": self.policy,     # pickle 可能(コミュニケータは除外される)
        # ... その他のホスト側の状態 ...
    }
    with open(filename, "wb") as f:
        pickle.dump(state, f)

    # 復元
    with open(filename, "rb") as f:
        state = pickle.load(f)
    self.policy = state["policy"]
    self.policy.set_comm(self.mpicomm)   # コミュニケータを再アタッチ

``set_comm()`` はコミュニケータの size と rank が保存された状態と一致
することを検証します。policy が Generator モード(``rng=`` 指定)で
あれば乱数状態は pickle に自動的に含まれます。legacy モードの場合は、
ホスト側で ``numpy.random.get_state()`` の保存・復元も併せて行って
ください(ODAT-SE は既にこれを行っています)。
