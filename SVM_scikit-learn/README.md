## サポートベクターマシン [SVM : Support Vector Machine]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [線形SVMによる３クラス（アヤメデータ）の識別問題](#線形SVMによる３クラス（アヤメデータ）の識別問題の実行結果)
    1. [RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題](#RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題)
    1. [RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題](#RBFカーネル関数を使用した３クラス（アヤメデータ）の識別問題)
    1. [RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題（その２）`main4.py`](#RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題（その２）)
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

>> 正解率の算出 : `sklearn.metrics.accuracy_score` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html </br>

>> SVM : `sklearn.svm.SVC` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


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
- モデルとして、線形SVMモデルを使用する :</br> 
`linearSVM = SVC( kernel = "linear", C = 1.0,  random_state = 0, probability = True )`
- この線形SVMモデルの fitting 処理でモデルを学習させる :</br>
`linearSVM.fit( X_train_std, y_train )`
- predict した結果 `y_predict = SVC.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `linearSVM.predict_proba( X_test_std[0, :].reshape(1, -1) )`

（※以下の挿入図のデータの分割方法の記載に、クロス・バリデーションの記載があるが、実際にはクロス・バリデーションによる各スコアの評価は行なっていない。）
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


<a id="RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題（その２）"></a>

## RBFカーネル関数を使用した２クラス（XORデータ分布）の識別問題（その２）`main4.py`
先の SVM によるXOR データの分類問題 `main2.py` において、グリッドサーチによるハイパーパラメータのチューニング、クロスバリデーションによる汎化性能の確認など、SVM を利用しての解析で、本来すべきことを追加したバージョン。<br>

![image](https://user-images.githubusercontent.com/25688193/49336445-a79c1300-f645-11e8-84a4-0b1b9a210b44.png)<br>
上図のように、`numpy.logical_xor()` で生成した、２つのクラス（-1 or 1）をとりうる２次元のXORで分布したデータ（特徴数 2 個 × サンプル数 200 個）に対して、C-SVMを用いて識別するタスクを考える。<br>
<br>
尚、この検証は、以下のような条件で行なっている。<br>
＜条件＞<br>
- XORデータ（サンプル数：２００個）<br>
- トレーニングデータ 80% 、テストデータ 20%の割合で分割<br>
- XORデータに対して、正規化処理実施<br>
- カーネル関数は、RBFカーネル<br>
- クロス・バディゲーション（k=10）で汎化性能を評価<br>

① グリッドサーチによるパラメーターのチューニング<br>
まずは、グリッドサーチを用いて、この分類タスクにおけるC-SVM の最適なパラメーター（＝C値、gamma値）をチューニングする。<br>
![image](https://user-images.githubusercontent.com/25688193/49336474-3872ee80-f646-11e8-8354-911587b76d31.png)<br>
上図は、scikit-learn の `sklearn.model_selection.GridSearchCV` モジュールを用いて、C-SVM のパラメーターをグリッドサーチし、対象のXOR データの正解率をヒートマップで図示したものである。（横軸と縦軸が、C-SVM のハイパーパラメータである RBFカーネルの gamma 値と、C-SVM の C 値）<br>
このヒートマップ図より、推定器として、カーネル関数を RBFカーネルとする C-SVM を使用した場合、最も正解率が高くなるパラメータ（ハイパーパラメータ）は、C = 1000, gamma = 0.1 (Accuracy = 0.971) となることが分かる。<br>
<br>
![image](https://user-images.githubusercontent.com/25688193/49336778-e0d78180-f64b-11e8-896d-b7520d8ca250.png)<br>

② 識別結果＆汎化性能の確認<br>
![image](https://user-images.githubusercontent.com/25688193/49336924-3dd43700-f64e-11e8-8fb2-c43cd8386057.png)<br>
上図は、グリッドサーチで最もよいスコアとなってC-SVM のハイパーパラメータ（C=1000,gamma=0.01）での、XOR データの識別境界の図である。<br>
ここで、○で囲んだ箇所は、テスト用データのサンプルであることを示している。<br>
<br>
![image](https://user-images.githubusercontent.com/25688193/49336922-2dbc5780-f64e-11e8-9082-fde0d1f19475.png)<br>


---