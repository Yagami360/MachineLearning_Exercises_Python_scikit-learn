## ロジスティクス回帰 [Logistic Regression]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [ロジスティクス回帰の実行結果](#ロジスティクス回帰の実行結果)
        1. [交差エントロピー関数の作図](#交差エントロピー関数の作図)
        1. [ロジスティクス回帰による３クラスの識別結果](#ロジスティクス回帰による３クラスの識別結果)
        1. [ロジスティクス回帰での正則化による過学習への対応](#ロジスティクス回帰での正則化による過学習への対応)
1. [背景理論](#背景理論)
    1. [ロジスティクス回帰によるパラメータ推定](#ロジスティクス回帰によるパラメータ推定)
    1. [最尤度法によるロジスティクス回帰モデルのパラメータ推定](#最尤度法によるロジスティクス回帰モデルのパラメータ推定)
    1. [ロジスティクス回帰による多クラスへの識別問題への拡張と、非線形変更、ガウス核関数](#ロジスティクス回帰による多クラスへの識別問題への拡張と、非線形変更、ガウス核関数)
    1. [ロジスティクス回帰での正則化による過学習への対応](#ロジスティクス回帰での正則化による過学習への対応)
    1. [【補足】汎化性能の評価](#汎化性能の評価)
        1. [バイアス・バリアントトレードオフ](#バイアス・バリアントトレードオフ)
        1. [過学習](#過学習)
        1. [正則化](#正則化)

<a name="#使用するライブラリ"></a>

### 使用するライブラリ

> scikit-learn ライブラリ </br>
>> データセット Dataset loading utilities : `sklearn.datasets`</br>
>> http://scikit-learn.org/stable/datasets/index.html </br>

>> モデル選択 : `sklearn.model_selection` </br>
>>> データの分割 : `sklearn.model_selection.train_test_split()`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html</br>

>> 前処理 : `sklearn.preprocessing` </br>
>>> 正規化処理 :  `sklearn.preprocessing.StandardScaler`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html </br>

>> 正解率の算出 : `sklearn.metrics.accuracy_score` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html </br>

>> ロジスティクス回帰 : `sklearn.linear_model.LogisticRegression` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : </br>
> https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#ロジスティクス回帰の実行結果"></a>

## ロジスティクス回帰の実行結果 : `main.py`

- Iris データセットを使用 : `datasets.load_iris()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 150 個 :</br> `iris.data[ :, [2,3] ]`
- サンプル数 150 個の内、品種 "setosa" が 50 個、"virginica" が 50 個、"versicolor" が 50 個。
- 教師データ `y_labels` は、サンプル数 150 個 : `y_labels = iris.target`
    - ここで、`iris.target` で取得したデータは、カテゴリーデータを 0, 1, 2 に変換したデータとなっている。
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施 : </br> `sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、ロジスティクス回帰モデルを使用する（L2正則化なし） :</br> 
`logReg = LogisticRegression( C = 100.0, random_state = 0 )`
- ロジスティクス回帰モデルの fitting 処理でモデルを学習させる。 :</br>
`logReg.fit( X_train_std, y_train )`
- predict した結果 `y_predict = logReg.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `logReg.predict_proba( X_test_std[0, :].reshape(1, -1) )`

</br>

<a name="#交差エントロピー関数のグラフの作図"></a>

### ① 交差エントロピー関数のグラフの作図
![twitter_ 18-17_170726](https://user-images.githubusercontent.com/25688193/29994417-b4362356-9009-11e7-9007-dc82d1ae793e.png)

<a name="#ロジスティクス回帰による３クラス識別結果"></a>

### ② ロジスティクス回帰による３クラス識別結果
（※以下の挿入図のデータの分割方法の記載に、クロス・バリデーションの記載があるが、実際にはクロス・バリデーションによる各スコアの評価は行なっていない。） 
![twitter_ 18-18_170726](https://user-images.githubusercontent.com/25688193/29994419-b440a164-9009-11e7-89ff-9fdb63fb537d.png)

- scikit-learn ライブラリの `sklearn.linear_model` モジュールの `LogisticRegression` クラスの`predict_proba()` 関数を使用して、指定したサンプルのクラスの所属確率を予想。
![logisticregression_scikit-learn_4](https://user-images.githubusercontent.com/25688193/28619864-725f0614-7245-11e7-8534-6c162eba8dd3.png)

<a name="#ロジスティクス回帰での正則化による過学習への対応"></a>

### ③ ロジスティクス回帰での正則化による過学習への対応
- 10個の逆正則化パラメータ C（C=10^-5,C=10^-4, ... C=10, ... , C=10^5 ）に関して、ロジスティクス回帰モデルを作成し、それぞれのモデルを学習データで学習。

```
    weights0 = []    # 重み係数の空リスト（Setosa）を生成
    weights1 = []    # 重み係数の空リスト（Versicolor）を生成
    weights2 = []    # 重み係数の空リスト（Virginica）を生成
    paramesC = []    # 逆正則化パラメータの空リストを生成

    # 10個の逆正則化パラメータ C（C=10^-5,C=10^-4, ... C=10, ... , C=10^5 ）に関して、
    # LogisiticReegression オブジェクトを作成し、それぞれのモデルを学習データで学習
    for c in numpy.arange(-5, 5):
        logReg_tmp = LogisticRegression.LogisticRegression( paramC = 10**c )
        logReg_tmp.logReg_.fit( X_train_std, y_train )

        # 重み係数リストに学習後の重み係数を格納
        # coef_[0] : petal length, petal weightの重み (Setosa)
        weights0.append( logReg_tmp.logReg_.coef_[0] )
        # coef_[1] : petal length, petal weightの重み (Versicolor)
        weights1.append( logReg_tmp.logReg_.coef_[1] )
        # coef_[2] : petal length, petal weightの重み (Virginica)   
        weights2.append( logReg_tmp.logReg_.coef_[2] )
        
        # 逆正則化パラメータリストにモデルの C 値を格納
        paramesC.append( 10**c )

    # 重み係数リストを numpy 配列に変換
    weights0 = numpy.array( weights0 )
    weights1 = numpy.array( weights1 )
    weights2 = numpy.array( weights2 )
```

- ロジスティクス回帰の逆正則化パラメータ C の値と正則化の強さの関係（ロジスティクス回帰における、正則化による過学習への対応）正則化の強さを確認するため、重み係数 w と逆正則化パラメータ C の関係を plot

![twitter_ 18-19_170727](https://user-images.githubusercontent.com/25688193/28652198-4b09b560-72c1-11e7-8053-a9e00b280ef8.png)

</br>

---

<a name="#背景理論"></a>

## 背景理論

<a name="#ロジスティクス回帰によるパラメータ推定"></a>

### ロジスティクス回帰によるパラメータ推定

![twitter_ 18-1_161130](https://user-images.githubusercontent.com/25688193/29994398-b3cb8b5e-9009-11e7-9ca3-947c8ede9407.png)
![twitter_ 18-2_161130](https://user-images.githubusercontent.com/25688193/29994397-b3ca7f84-9009-11e7-8e86-9677931b681e.png)
![twitter_ 18-3_161130](https://user-images.githubusercontent.com/25688193/29994396-b3c9dcd2-9009-11e7-8db0-c342aac2725c.png)
![twitter_ 18-4_161130](https://user-images.githubusercontent.com/25688193/29994399-b3cb73f8-9009-11e7-8f86-52d112491644.png)
![twitter_ 18-5_161201](https://user-images.githubusercontent.com/25688193/29994401-b3ceb5d6-9009-11e7-97b6-9470f10d0235.png)

<a name="#最尤度法によるロジスティクス回帰モデルのパラメータ推定"></a>

### 最尤度法によるロジスティクス回帰モデルのパラメータ推定 [MLE]
![twitter_ 18-6_161201](https://user-images.githubusercontent.com/25688193/29994400-b3cdbcf8-9009-11e7-9dba-fdaf84d592f8.png)
![twitter_ 18-6 _170204](https://user-images.githubusercontent.com/25688193/29994403-b3ed4870-9009-11e7-8432-0468dfc2b841.png)
![twitter_ 18-7_161201](https://user-images.githubusercontent.com/25688193/29994405-b3ee6e94-9009-11e7-840d-50d2a5c10aba.png)
![twitter_ 18-7 _170204](https://user-images.githubusercontent.com/25688193/29994406-b3efd13a-9009-11e7-817d-6f0d5373f178.png)

<a name="#ロジスティクス回帰による多クラスの識別問題への拡張と、非線形変更、ガウス核関数"></a>

### ロジスティクス回帰による多クラスの識別問題への拡張と、非線形変更、ガウス核関数
![twitter_ 18-8_170204](https://user-images.githubusercontent.com/25688193/29994404-b3ee8d34-9009-11e7-8866-675b5083222e.png)
![twitter_ 18-9_170208](https://user-images.githubusercontent.com/25688193/29994407-b3f1c864-9009-11e7-8b50-b0da25938bc7.png)
![twitter_ 18-10_170208](https://user-images.githubusercontent.com/25688193/29994408-b3f4b9ac-9009-11e7-8d49-e639ecadc702.png)
![twitter_ 18-11_170209](https://user-images.githubusercontent.com/25688193/29994410-b4128f40-9009-11e7-8cd9-27f9f29d1be3.png)
![twitter_ 18-11 _170210](https://user-images.githubusercontent.com/25688193/29994409-b410782c-9009-11e7-9cc6-743895e9af2a.png)
![twitter_ 18-12_170209](https://user-images.githubusercontent.com/25688193/29994411-b413263a-9009-11e7-990b-8ac7a180ecba.png)
![twitter_ 18-12 _170210](https://user-images.githubusercontent.com/25688193/29994413-b4180088-9009-11e7-9020-33405676bbee.png)
![twitter_ 18-13_170210](https://user-images.githubusercontent.com/25688193/29994412-b416731c-9009-11e7-88fc-43309099b794.png)
![twitter_ 18-14_170210](https://user-images.githubusercontent.com/25688193/29994414-b41e7fa8-9009-11e7-8333-56962c5a82b8.png)
![twitter_ 18-15_170210](https://user-images.githubusercontent.com/25688193/29994415-b432a4ec-9009-11e7-8bf2-cd7cdd0f42e1.png)
![twitter_ 18-16_170210](https://user-images.githubusercontent.com/25688193/29994416-b4360ca4-9009-11e7-8470-a16e8f0c648d.png)

<a name="#汎化性能の評価"></a>

### 【補足】汎化性能の評価

![twitter_ _3-1_160924](https://user-images.githubusercontent.com/25688193/29994477-179abe2e-900b-11e7-99ac-0691ba52e2d2.png)
![twitter_ _3-2_160924](https://user-images.githubusercontent.com/25688193/29994475-179a835a-900b-11e7-8e20-d2f893d340f7.png)
![twitter_ _3-3_160924](https://user-images.githubusercontent.com/25688193/29994473-1799ebf2-900b-11e7-882e-fe29fb378f72.png)
![twitter_ _3-4_170727](https://user-images.githubusercontent.com/25688193/29994474-179a4d40-900b-11e7-98d9-0ab3825217ba.png)
![twitter_ _3-5_170810](https://user-images.githubusercontent.com/25688193/29994478-179b2116-900b-11e7-80cf-513939ff6822.png)
![twitter_ _3-6_170810](https://user-images.githubusercontent.com/25688193/29994476-179aaa60-900b-11e7-9d72-397ddf43eb9e.png)
![twitter_ _3-7_170810](https://user-images.githubusercontent.com/25688193/29994479-17bc86c6-900b-11e7-93b6-d5e1c56b8e48.png)
![twitter_ _3-8_170810](https://user-images.githubusercontent.com/25688193/29994480-17bcad68-900b-11e7-9e60-4a1bc9494a27.png)

