## ロジスティクス回帰 [Logistic Regression]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [線形SVMによる３クラス（アヤメデータ）の識別問題](#線形SVMによる３クラス（アヤメデータ）の識別問題の実行結果)
    1. [RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題](#RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題)
    1. [RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題](#RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題)
1. [背景理論](#背景理論)
    1. [サポートベクターマシンの概要](#サポートベクターマシンの概要)
    1. [マージンとマージン最大化](#マージンとマージン最大化)
    1. [マージン最大化と凸最適化問題](#マージン最大化と凸最適化問題)
        1. [KTT条件](#KTT条件)
    1. [線形分離不可能な系への拡張とソフトマージン識別器、スラック変数の導入](#線形分離不可能な系への拡張とソフトマージン識別器、スラック変数の導入)
    1. [サポートベクターマシンにおける非線形特徴写像（カーネル関数、カーネル法、カーネルトリック）](#サポートベクターマシンにおける非線形特徴写像（カーネル関数、カーネル法、カーネルトリック）)
    1. [カーネル関数](#カーネル関数)
        1. [多項式カーネル関数](#多項式カーネル関数)
        1. [動径基底カーネル関数](#動径基底カーネル関数)

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

>> SVM : `sklearn.svm.SVC` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

>> 正解率の算出 : `sklearn.metrics.accuracy_score` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html </br>


</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : </br>
> https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

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
- モデルとして、線形SVMモデルを使用する :</br> 
`linearSVM = SVC( kernel = "linear", C = 1.0,  random_state = 0, probability = True )`
- この線形SVMモデルの fitting 処理でモデルを学習させる :</br>
`linearSVM.fit( X_train_std, y_train )`
- predict した結果 `y_predict = SVC.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `linearSVM.predict_proba( X_test_std[0, :].reshape(1, -1) )`

![twitter_svm_6-1_170728](https://user-images.githubusercontent.com/25688193/28708061-adc48348-73b5-11e7-8cf8-17f3c3a8ba0e.png)

## RBFカーネル関数を使用した２クラス識別問題の実行結果 : `main2.py`

- XOR なデータセットを使用 : `numpy.logical_xor()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 200 個 の正規分布に従う乱数:</br> `X_features = numpy.random.randn( 200, 2 )`
- 特徴行列を XORした結果でクラス分け : </br> 
`y_labels = numpy.logical_xor( (X_features[:,0] > 0), (X_features[:,1] > 0) )`
- クラス分けしたラベルデータ `y_labels` を -1, 1 に変換 : </br>
`numpy.where( y_labels > 0 , 1, -1 )`
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施 : </br> 
`sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、カーネル関数を RBF カーネル関数とする、C-SVM モデルを使用する :</br> 
`kernelSVM1  = SVC( kernel = 'rbf', C = 10.0,  gamma = 0.10, random_state = 0, probability = True )`
- この C-SVM モデルの fitting 処理でモデルを学習させる :</br>
`kernelSVM1.fit( X_train_std, y_train )`
- predict した結果 `y_predict = SVC.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `kernelSVM1.predict_proba( X_test_std[0, :].reshape(1, -1) )`

![twitter_svm_6-2 _170728](https://user-images.githubusercontent.com/25688193/28719743-f71ebd8a-73e5-11e7-91cb-476014748aad.png)

## RBFカーネル関数を使用した３クラス（アヤメデータ）識別問題の実行結果 : `main3.py`

- Iris データセットを使用 : `datasets.load_iris()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 150 個 :</br> `iris.data[ :, [2,3] ]`
- サンプル数 150 個の内、品種 "setosa" が 50 個、"virginica" が 50 個、"versicolor" が 50 個。
- 教師データ `y_labels` は、サンプル数 150 個 : `y_labels = iris.target`
    - ここで、`iris.target` で取得したデータは、カテゴリーデータを 0, 1, 2 に変換したデータとなっている。
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施 : </br> 
`sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、カーネル関数を RBF カーネル関数とする、それぞれハイパーパラメータの異なる３つの C-SVM モデルを使用する :</br> 
`kernelSVM1  = SVC( kernel = 'rbf', C = 1.0,  gamma = 0.20, random_state = 0, probability = True )`</br>
`kernelSVM2  = SVC( kernel = 'rbf', C = 10.0,  gamma = 0.20, random_state = 0, probability = True )`</br>
`kernelSVM3  = SVC( kernel = 'rbf', C = 1.0,  gamma = 100.0, random_state = 0, probability = True )`</br>
- それらの C-SVM モデルの fitting 処理でモデルを学習させる :</br>
`kernelSVM1.fit( X_train_std, y_train )`</br>
`kernelSVM2.fit( X_train_std, y_train )`</br>
`kernelSVM3.fit( X_train_std, y_train )`</br>
- predict した結果 `y_predict = SVC.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `kernelSVM1.predict_proba( X_test_std[0, :].reshape(1, -1) )`

![twitter_svm_6-3_170729](https://user-images.githubusercontent.com/25688193/28736123-694aa3d2-7423-11e7-8bba-92fadfdc645c.png)
![twitter_svm_6-4_170729](https://user-images.githubusercontent.com/25688193/28737648-6f478f8c-742a-11e7-9de9-f3f6d619d623.png)

---

<a name="#背景理論"></a>

## 背景理論

<a name="#サポートベクターマシンの概要"></a>

### サポートベクターマシンの概要

![twitter_svm_1-1_170211](https://user-images.githubusercontent.com/25688193/28708179-313cdc98-73b6-11e7-985f-8ced8d316ecc.png)

<a name="#マージンとマージン最大化"></a>

### マージンとマージン最大化
![twitter_svm_1-2_170211](https://user-images.githubusercontent.com/25688193/28708178-313a6daa-73b6-11e7-9817-8621f3cd9985.png)
![twitter_svm_2-1_170212](https://user-images.githubusercontent.com/25688193/28708177-31342c92-73b6-11e7-9b19-0a41a4b7b705.png)
![twitter_svm_2-2_170212](https://user-images.githubusercontent.com/25688193/28708175-312ab5c2-73b6-11e7-8617-37b57c475b35.png)

<a name="#マージン最大化と凸最適化問題"></a>

### マージン最大化と凸最適化問題
![twitter_svm_3-1_170214](https://user-images.githubusercontent.com/25688193/28708174-311e33d8-73b6-11e7-82e5-3da320e93b89.png)
![twitter_svm_3-2_170214](https://user-images.githubusercontent.com/25688193/28708173-311dbda4-73b6-11e7-832e-bf7162703056.png)
![twitter_svm_3-3_170214](https://user-images.githubusercontent.com/25688193/28708172-3118eeaa-73b6-11e7-960a-71824390bee5.png)
![twitter_svm_3-4_170214](https://user-images.githubusercontent.com/25688193/28708171-3113dc62-73b6-11e7-9140-f4974f44b495.png)

<a name="#KKT条件"></a>

### KKT [Karush-Kuhn-Tucker] 条件
![twitter_svm_3-5_170216](https://user-images.githubusercontent.com/25688193/28708170-31097290-73b6-11e7-8d0c-8087e1751fb1.png)

<a name="#線形分離不可能な系への拡張とソフトマージン識別器、スラック変数の導入"></a>

### 線形分離不可能な系への拡張とソフトマージン識別器、スラック変数の導入
![twitter_svm_4-1_170216](https://user-images.githubusercontent.com/25688193/28708169-310200aa-73b6-11e7-8492-41e07ad0a3f9.png)
![twitter_svm_4-2_170217](https://user-images.githubusercontent.com/25688193/28708168-30faf92c-73b6-11e7-987b-996e874fb16f.png)
![twitter_svm_4-3_170217](https://user-images.githubusercontent.com/25688193/28708165-30eb1a5c-73b6-11e7-8530-e19ac4cef9e1.png)
![twitter_svm_4-4_170218](https://user-images.githubusercontent.com/25688193/28708167-30f916ac-73b6-11e7-976d-d4c1e3a52524.png)
![twitter_svm_4-5_170218](https://user-images.githubusercontent.com/25688193/28708166-30f5c588-73b6-11e7-9d9b-54d46b8a69f5.png)

<a name="#サポートベクターマシンにおける非線形特徴写像（カーネル関数、カーネル法、カーネルトリック）"></a>

### サポートベクターマシンにおける非線形特徴写像（カーネル関数、カーネル法、カーネルトリック）
![twitter_svm_5-1_170219](https://user-images.githubusercontent.com/25688193/28708164-30e4d688-73b6-11e7-89a0-d78b5065b467.png)
![twitter_svm_5-2_170220](https://user-images.githubusercontent.com/25688193/28708163-30def074-73b6-11e7-8d17-57fdbf9bab59.png)
![twitter_svm_5-2_170225](https://user-images.githubusercontent.com/25688193/28708162-30c28aba-73b6-11e7-8e63-aa1d77db8c00.png)
![twitter_svm_5-3_170222](https://user-images.githubusercontent.com/25688193/28708159-30bd4c44-73b6-11e7-91bb-c212ab04a7db.png)

<a name="#多項式カーネル関数"></a>

### 多項式カーネル関数
![twitter_svm_5-4_170225](https://user-images.githubusercontent.com/25688193/28708161-30c06262-73b6-11e7-88bd-9ea72837d9c8.png)

<a name="#動径基底カーネル関数"></a>

### 動径基底カーネル [RBF-kernel] 関数
![twitter_svm_5-5_170303](https://user-images.githubusercontent.com/25688193/28708158-30bc0e1a-73b6-11e7-9fc1-c015e9403def.png)
![twitter_svm_5-6_170303](https://user-images.githubusercontent.com/25688193/28708157-30bbfba0-73b6-11e7-9aba-894974b30167.png)

