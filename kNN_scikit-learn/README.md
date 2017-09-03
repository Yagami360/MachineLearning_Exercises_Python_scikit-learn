## k-NN 法 [k-nearest neighbor algorithm]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [線形SVMによる３クラス（アヤメデータ）の識別問題](#線形SVMによる３クラス（アヤメデータ）の識別問題の実行結果)
    1. [RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題](#RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題)
    1. [RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題](#RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題)
1. [背景理論](#背景理論)
    1. [最近傍法、k-NN 法の概要](#最近傍法、k-NN法の概要)
    1. [最近傍法とボロノイ図](#最近傍法とボロノイ図)
        1. [鋳型の数と識別性能](#鋳型の数と識別性能)
    1. [k-NN 法](#k-NN法)
        1. [k-NN 法での誤り発生のメカニズムとベイズの誤り率の関係](#k-NN法での誤り発生のメカニズムとベイズの誤り率の関係)

</br>

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

>> k-NN 法 : `sklearn.neighbors.KNeighborsClassifier` </br>
sklearn.neighbors.KNeighborsClassifier


</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : `datasets.load_iris()`

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#線形SVMによる３クラス（アヤメデータ）の識別問題"></a>

## 線形SVMによる３クラス（アヤメデータ）の識別問題 : `main1.py`

- Iris データセットを使用 : `datasets.load_iris()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 150 個 :</br> `iris.data[ :, [2,3] ]`
- サンプル数 150 個の内、品種 "setosa" が 50 個、"virginica" が 50 個、"versicolor" が 50 個。
- 教師データ `y_labels` は、サンプル数 150 個 : `y_labels = iris.target`
    - ここで、`iris.target` で取得したデータは、カテゴリーデータを 0, 1, 2 に変換したデータとなっている。
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施 : </br> 
`sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、それぞれハイパーパラメータの異なる３つの k-NN モデルを使用する。</br> （グリッドサーチによる最適なハイパーパラメータの検討は行わない）:</br> 
`kNN1 = KNeighborsClassifier( n_neighbors = 1, p = 2, metric = 'minkowski' )`</br>
`kNN2 = KNeighborsClassifier( n_neighbors = 5, p = 2, metric = 'minkowski' )`</br>
`kNN3 = KNeighborsClassifier( n_neighbors = 10, p = 2, metric = 'minkowski' )`</br>
- それらの k-NN モデルの fitting 処理でモデルを学習させる :</br>
`kNN1.fit( X_train_std, y_train )`</br>
`kNN2.fit( X_train_std, y_train )`</br>
`kNN3.fit( X_train_std, y_train )`</br>
- predict した結果 `y_predict = kNN.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `kNN1.predict_proba( X_test_std[0, :].reshape(1, -1) )`

![twitter_ 16-7_170729](https://user-images.githubusercontent.com/25688193/28742632-1482008c-7470-11e7-9590-df87069db4ed.png)

---

<a name="#背景理論"></a>

## 背景理論

<a name="#最近傍法、k-NN法の概要"></a>

### 最近傍法、k-NN 法の概要

![twitter_ 14-1_161007](https://user-images.githubusercontent.com/25688193/28742174-1d0f13f4-7464-11e7-8cc9-1d669f2c50ca.png)

<a name="#最近傍法とボロノイ図"></a>

### 最近傍法とボロノイ図
![twitter_ 14-2_161007](https://user-images.githubusercontent.com/25688193/28742169-1d0c9bce-7464-11e7-97c2-0ec640aa3e15.png)
![twitter_ 14-3_161008](https://user-images.githubusercontent.com/25688193/28742170-1d0d1270-7464-11e7-8cfb-5ec25983427f.png)
![twitter_ 14-4_161009](https://user-images.githubusercontent.com/25688193/28742171-1d0e1530-7464-11e7-8e32-04b007727098.png)

<a name="#鋳型の数と識別性能"></a>

### 鋳型の数と識別性能
![twitter_ 14-5_161010](https://user-images.githubusercontent.com/25688193/28742173-1d0f097c-7464-11e7-8df7-cd6018620fbf.png)

<a name="#k-NN法"></a>

### k-NN 法
![twitter_ 16-1_161011](https://user-images.githubusercontent.com/25688193/28742172-1d0edbfa-7464-11e7-8e82-238a91faf92e.png)
![twitter_ 16-2_161012](https://user-images.githubusercontent.com/25688193/28742176-1d2fe52a-7464-11e7-825d-6d49ca8ccfed.png)

<a name="#k-NN法での誤り発生のメカニズムとベイズの誤り率の関係"></a>

### k-NN 法での誤り発生のメカニズムとベイズの誤り率の関係
![twitter_ 16-5_161112](https://user-images.githubusercontent.com/25688193/28742175-1d2f1b0e-7464-11e7-9b18-3d74ddd6e142.png)
![twitter_ 16-6_161112](https://user-images.githubusercontent.com/25688193/28742177-1d31eb68-7464-11e7-8bd6-a9443593c392.png)
