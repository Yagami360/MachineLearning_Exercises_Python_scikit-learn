## パーセプトロン [Perceptron]（scikit-learn ライブラリ使用）

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [パーセプトロンによる３クラス（アヤメデータ）の識別](#パーセプトロンによる３クラス（アヤメデータ）の識別)
    1. [](#)
1. [背景理論](#背景理論)
    1. [ニューラルネットワークの概要](#ニューラルネットの概要)
    1. [活性化関数](#活性化関数)
    1. [単純パーセプトロン](#単純パーセプトロン)
    1. [パーセプトロンによる論理演算](#パーセプトロンによる論理演算)
    1. [最急降下法による学習](#最急降下法による学習)

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

>> パーセプトロン : `sklearn.linear_model.Perceptron` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

>> 正解率の算出 : `sklearn.metrics.accuracy_score` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html </br>


</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : `datasets.load_iris()`

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#パーセプトロンによる３クラス（アヤメデータ）の識別"></a>

## パーセプトロンによる３クラス（アヤメデータ）の識別 : `main.py`

- Iris データセットを使用 : `datasets.load_iris()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 150 個 :</br> `iris.data[ :, [2,3] ]`
- サンプル数 150 個の内、品種 "setosa" が 50 個、"virginica" が 50 個、"versicolor" が 50 個。
- 教師データ `y_labels` は、サンプル数 150 個 : `y_labels = iris.target`
    - ここで、`iris.target` で取得したデータは、カテゴリーデータを 0, 1, 2 に変換したデータとなっている。
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施する。</br> : `sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、単純パーセプトロンモデルを使用する :</br> 
`ppn1 = Perceptron( n_iter = 40, eta0 = 0.1, random_state = 0, shuffle = True )`
- この線形SVMモデルの fitting 処理でモデルを学習させる :</br>
`ppn1.fit( X = X_train_std, y = y_train )`
- predict した結果 `y_predict = ppn1.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `linearSVM.predict_proba( X_test_std[0, :].reshape(1, -1) )`

</br>

パーセプトロンによる３クラス（アヤメデータ）の識別結果と識別境界の図。 </br>
（重みの更新:Δw = η*(y_i-y^_i)）</br>
※ 以下の図中の分割方法に関する記述の、クロス・バリデーションの部分は誤記で、クロス・バリデーションでの各種スコアの評価は行なっていない。
![twitter_python_scikit-learn_1_1_170719](https://user-images.githubusercontent.com/25688193/29999283-590ac8dc-907d-11e7-8202-b61ca7134164.png)

</br>

---

<a name="#背景理論"></a>

## 背景理論

<a name="#ニューラルネットワークの概要"></a>

## ニューラルネットワークの概要
![twitter_nn1_1_160825](https://user-images.githubusercontent.com/25688193/29994077-594d50c4-9002-11e7-829d-5a695503b486.png)
![twitter_nn1_2_160825](https://user-images.githubusercontent.com/25688193/29994078-594deebc-9002-11e7-801f-d0d6617cbde6.png)
![twitter_nn3 -1_160827](https://user-images.githubusercontent.com/25688193/29994081-5976f6cc-9002-11e7-9587-dc3cb098b325.png)

<a name="#活性化関数"></a>

## 活性化関数
![twitter_nn2-1_160826](https://user-images.githubusercontent.com/25688193/29994079-59705a74-9002-11e7-88ba-214af1ceec62.png)
![twitter_nn2-2_160826](https://user-images.githubusercontent.com/25688193/29994080-5970ebe2-9002-11e7-86fb-769349356224.png)

<a name="#単純パーセプトロン"></a>

# 単純パーセプトロン
![twitter_nn4 -1_160829](https://user-images.githubusercontent.com/25688193/29994084-598c65c0-9002-11e7-9f9b-a529d44f1f8a.png)

<a name="#パーセプトロンによる論理演算"></a>

# パーセプトロンによる論理演算
![twitter_nn6-1_160829](https://user-images.githubusercontent.com/25688193/29994082-597791ea-9002-11e7-9bb5-2ae6bc436f56.png)
![twitter_nn6-2_160829](https://user-images.githubusercontent.com/25688193/29994083-598aa280-9002-11e7-9ec0-16316a04686a.png)

<a name="#最急降下法による学習"></a>

# 最急降下法による学習
![twitter_nn8-2 _160902](https://user-images.githubusercontent.com/25688193/29994085-59937f04-9002-11e7-974e-a9cd6fa61f13.png)
![twitter_nn8-3 _160902](https://user-images.githubusercontent.com/25688193/29994086-5997cc9e-9002-11e7-87e8-1ab817704a8a.png)
