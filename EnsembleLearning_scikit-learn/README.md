## アンサンブルモデルとアンサンブル学習 [Ensemble Learning]

コード実装中...

### 目次 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [多数決方式のアンサンブル法と、単体での分類器での誤分類率。及び多数決方式のアンサンブル法における分類器の個数に応じた比較 : `main1.py`](#main1.py)
    1. [多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ : `main2.py`](#main2.py)
1. [背景理論](#背景理論)
    1. [混合モデルとアンサンブル学習](#混合モデルとアンサンブル学習)
    1. [決定木](#決定木)
    1. [アダブースト](#アダブースト)
    1. [バギング](#バギング)
    1. [ランダムフォレスト](#ランダムフォレスト)
    1. [EMアルゴリズム](#EMアルゴリズム)

<a name="#使用するライブラリ"></a>

### 使用するライブラリ：

> scikit-learn ライブラリ</br>
>> 開発者向け情報 : </br>
http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
>> `sklearn.base` モジュールの API Reference
>>> `sklearn.base` :</br>
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base
>>> `sklearn.base.BaseEstimator` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator

> その他ライブラリ
>> `math` :</br>
https://docs.python.org/2/library/math.html
>> `scipy.misc.comb` :</br>
https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.comb.html


<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データ : `sklearn.datasets.load_iris()`</br>
>>

|行番号|1|2|3|4|5|
|---|---|---|---|---|---|
|0||||||
|1||||||

> 同心円状のデータセット : `sklearn.datasets.make_circles()` </br>
> 半月状のデータセット : `sklearn.datasets.make_moons()` </br>

---

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#main1.py"></a>

### 多数決方式のアンサンブル法と、単体での分類器での誤分類率。及び多数決方式のアンサンブル法における分類器の個数に応じた比較 : </br> `main1.py`

>多数決方式のアンサンブル法（最終的な識別結果を複数の分類器での多数決で決め、２項分布の累積に従う）と、単体での分類器での誤分類率の比較図、及び多数決方式のアンサンブル法における分類器の個数に応じた比較図。</br>
分類器の個数が奇数で、ランダムな結果（0.5）より識別性能が高い（＝図では 0.5 より小さい領域）場合、多数決方式でのアンサンブルな手法のほうが、単体での分類器より、常に誤識別が低い（識別性能が高い）ことが分かる。</br>
尚、分類器の個数が偶数個の場合は、必ずしもこれが成り立つとは限らないことに注意（この多数決方式のアンサンブル法では多数決により最終予想を決めるため。）

![ensemblelearning_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29705020-33fd8704-89b7-11e7-9760-5d04bca26af6.png)


<a name="#main2.py"></a>

### 多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ : </br> `main2.py`

- 検証用データとして、Iris データセットを使用
  - トレーニングデータ 50% 、テストデータ 50%の割合で分割 :</br> `sklearn.model_selection.train_test_split()` を使用
- アンサンブル法による分類器の自作クラス `EnsembleLearningClassifier` を使用
  - この自作クラス `EnsembleLearningClassifier` に scikit -learn ライブラリ の `Pipeline` クラスを設定
    - １つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`
    - １つ目のパイプラインの推定器は、ロジスティクス回帰 : </br>`( "clf", LogisticRegression( penalty = 'l2', C = 0.001, random_state = 0 )`
    - ２つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`
    - ２つ目のパイプラインの推定器は、決定木 : </br>`( "clf", DecisionTreeClassifier( max_depth = 3,  criterion = 'entropy', random_state = 0 )`
    - ３つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`
    - ３つ目のパイプラインの推定器は、k-NN 法 : </br>`( "clf", KNeighborsClassifier( n_neighbors = 3, p = 2, metric = 'minkowski' )`
- クロス・バディゲーション k-fold CV (k=10) で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score()` を使用

#### Iris データセット（トレーニングデータ 50% 、テストデータ 50%の割合で分割）での検証結果

> 各種スコア値 by k-fold CV (cv=10) :（チューニング前）

|Model (classifiers)|Accuracy</br>[train data]|Accuracy</br>[test data]|AUC</br>[train data]|AUC</br>[test data]|
|---|---|---|---|---|
|Logistic Regression </br> `penalty = 'l2', C = 0.001`|0.84 (+/- 0.23)|0.83 (+/- 0.18)|0.92 (+/- 0.20)|0.95 (+/- 0.11)|
|Decision Tree </br> `criterion = 'entropy', max_depth = 3`|0.92 (+/- 0.13)|0.81 (+/- 0.16)|0.92 (+/- 0.15)|0.85 (+/- 0.14)|
|k-NN </br> `n_neighbors = 3, metric = 'minkowski'`|0.92 (+/- 0.13)|0.83 (+/- 0.14)|0.93 (+/- 0.12)|0.89 (+/- 0.11)|
|SVM</br> ` kernel = 'rbf', C = 0.50, gamma = 0.10 `|0.88 (+/- 0.14)|0.90 (+/- 0.10)|0.95 (+/- 0.15)|0.98 (+/- 0.05)|
|Ensemble Model 1</br> [LogisticRegression, DecisionTree, k-NN]|...|...|...|...|
|Ensemble Model 2</br> [LogisticRegression, DecisionTree, SVM]|...|...|...|...|

</br>

> 識別結果＆識別境界（チューニング前）
>> チューニング前の適当なハイパーパラメータでの各弱識別器＆これらのアンサンブルモデルでの識別結果＆識別境界の図。上段の図がトレーニングデータでの結果。下段がテストデータでの結果。アンサンブルモデルでは、これを構成する個々の弱識別器の識別境界を混ぜ合わせた形状になっていることが分かる。

![ensemblelearning_scikit-learn_2-1](https://user-images.githubusercontent.com/25688193/29748312-3d019e96-8b4f-11e7-9ee1-845eba6df331.png)

![ensemblelearning_scikit-learn_2-2](https://user-images.githubusercontent.com/25688193/29748309-2e737764-8b4f-11e7-8bc6-e62332dd0c8b.png)

</br>

> 学習曲線（チューニング前）

![ensemblelearning_scikit-learn_3-1](https://user-images.githubusercontent.com/25688193/29748314-4b14c260-8b4f-11e7-9ee5-4bcc9ce98ece.png)

![ensemblelearning_scikit-learn_3-2](https://user-images.githubusercontent.com/25688193/29748316-549b23c4-8b4f-11e7-9542-6c3484487bdf.png)

</br>

> ROC 曲線（チューニング前）

</br>

> グリッドサーチによる各弱識別器のチューニング

</br>

> 各モデルでの識別境界
>> コード実施中...

</br>

#### 同心円状データセットでの検証結果

> 各種スコア値 by k-fold CV : `cross_val_score( cv = 10 )`
>> コード実施中...

|Model|Accuracy</br>[train data]|Accuracy</br>[test data]|AUC</br>[train data]|AUC</br>[test data]|
|---|---|---|---|---|
|Logistic Regression </br> `penalty = 'l2', C = 0.001`|||
|Decision Tree </br> `criterion = 'entropy', max_depth = 3`|||
|k-NN </br> `n_neighbors = 3, metric='minkowski'`|||
|SVM </br> ``|||
|Ensemble 1</br>[LogisticRegression, DecisionTree, k-NN]|||

> 各モデルでの識別境界
>> コード実施中...

---

<a name="#背景理論"></a>

## 背景理論

<a name="#混合モデルとアンサンブル学習"></a>

## 混合モデルとアンサンブル学習

![twitter_ _ _11-1_170626](https://user-images.githubusercontent.com/25688193/29602474-4747a2b0-881b-11e7-9881-f79b0d22a1f2.png)
![twitter_ _ _11-2_170630](https://user-images.githubusercontent.com/25688193/29602475-4747ec34-881b-11e7-841a-bf69defa1311.png)
![twitter_ _ _11-3_170630](https://user-images.githubusercontent.com/25688193/29602473-47477d3a-881b-11e7-9d38-75561216b8a3.png)
![twitter_ _ _11-4_170630](https://user-images.githubusercontent.com/25688193/29602476-47481830-881b-11e7-8f8a-dfe9e358ff9e.png)
![twitter_ _ _11-5_170630](https://user-images.githubusercontent.com/25688193/29602477-47481d30-881b-11e7-9162-3e9101f12ca1.png)

<a name="#決定木"></a>

## 決定木

![twitter_ 21-1_170730](https://user-images.githubusercontent.com/25688193/29602731-5f1cc8b0-881c-11e7-89ae-bdc9d923d3a8.png)
![twitter_ 21-2_170730](https://user-images.githubusercontent.com/25688193/29602727-5f08c57c-881c-11e7-9e3c-f2c4736d4ed4.png)
![twitter_ 21-3_170731](https://user-images.githubusercontent.com/25688193/29602729-5f1b7a32-881c-11e7-8e6f-67b17765d140.png)
![twitter_ 21-5_170731](https://user-images.githubusercontent.com/25688193/29602728-5f08efde-881c-11e7-8688-f44bdcf2f675.png)
![twitter_ 21-6_170731](https://user-images.githubusercontent.com/25688193/29602730-5f1cc036-881c-11e7-96b9-b4a5a08d016e.png)
![twitter_ 21-7_170731](https://user-images.githubusercontent.com/25688193/29602732-5f1d5398-881c-11e7-8efa-d4d5a2abb7c6.png)
![twitter_ 21-8_170801](https://user-images.githubusercontent.com/25688193/29602734-5f39ad7c-881c-11e7-8007-14ed56644f94.png)
![twitter_ 21-9_170801](https://user-images.githubusercontent.com/25688193/29602733-5f2bfb78-881c-11e7-8906-7f9935fdbea2.png)
![twitter_ 21-10_170801](https://user-images.githubusercontent.com/25688193/29602735-5f3eba38-881c-11e7-898f-94e586787fe8.png)
![twitter_ 21-11_170801](https://user-images.githubusercontent.com/25688193/29602737-5f4051b8-881c-11e7-81c2-83d7b0b6f74f.png)
![twitter_ 21-12_170802](https://user-images.githubusercontent.com/25688193/29602736-5f401284-881c-11e7-8210-bfac76aa1cc3.png)
![twitter_ 21-13_170802](https://user-images.githubusercontent.com/25688193/29602738-5f4182c2-881c-11e7-8856-a295bdc0161e.png)
![twitter_ 21-14_170802](https://user-images.githubusercontent.com/25688193/29602739-5f4f9830-881c-11e7-87cb-a79d65401d31.png)
![twitter_ 21-15_170802](https://user-images.githubusercontent.com/25688193/29602740-5f60cd12-881c-11e7-9a49-894709d818b4.png)


<a name="#アダブースト"></a>

## アダブースト

![twitter_ _ _11-11_170702](https://user-images.githubusercontent.com/25688193/29602479-476bbe66-881b-11e7-9927-231b8268982a.png)
![twitter_ _ _11-12_170703](https://user-images.githubusercontent.com/25688193/29602484-478cfc48-881b-11e7-88b0-fecbc07b4ab7.png)
![twitter_ _ _11-13_170703](https://user-images.githubusercontent.com/25688193/29602486-4791bf94-881b-11e7-8156-7a92cef0a55f.png)
![twitter_ _ _11-14_170704](https://user-images.githubusercontent.com/25688193/29602487-47937bc2-881b-11e7-905b-1e4e5d9620e8.png)
![twitter_ _ _11-15_170704](https://user-images.githubusercontent.com/25688193/29602485-479175fc-881b-11e7-9bbd-e2ce8193b60f.png)
![twitter_ _ _11-16_170705](https://user-images.githubusercontent.com/25688193/29602490-47b3f3e8-881b-11e7-98f2-f442760ab208.png)
![twitter_ _ _11-17_170705](https://user-images.githubusercontent.com/25688193/29602488-47947fa4-881b-11e7-94f7-c2fc9b06a58c.png)

<a name="#バギング"></a>

## バギング

![twitter_ _ _11-18_170705](https://user-images.githubusercontent.com/25688193/29602489-47b34c9a-881b-11e7-80d7-b34afe348feb.png)
![twitter_ _ _11-19_170707](https://user-images.githubusercontent.com/25688193/29602491-47c72a08-881b-11e7-82df-71b94ae2dfc2.png)

<a name="#ランダムフォレスト"></a>

## ランダムフォレスト

![twitter_ 22-1_170802](https://user-images.githubusercontent.com/25688193/29602642-f67869f4-881b-11e7-9752-d37b693461e6.png)
![twitter_ 22-2_170802](https://user-images.githubusercontent.com/25688193/29602644-f69bf234-881b-11e7-9142-976f7294ee28.png)
![twitter_ 22-3_170802](https://user-images.githubusercontent.com/25688193/29602643-f69b541e-881b-11e7-8bf7-506da7bd0676.png)
![twitter_ 22-4_170803](https://user-images.githubusercontent.com/25688193/29602646-f69dc32a-881b-11e7-8b21-4af806db4294.png)
![twitter_ 22-5_170803](https://user-images.githubusercontent.com/25688193/29602648-f6b0a30a-881b-11e7-85cf-13ba4ced8060.png)
![twitter_ 22-6_170804](https://user-images.githubusercontent.com/25688193/29602647-f6b054f4-881b-11e7-8925-d874f1ff7e0e.png)
![twitter_ 22-7_170804](https://user-images.githubusercontent.com/25688193/29602645-f69dae12-881b-11e7-9d77-7342fc3aac86.png)
![twitter_ 22-8_170804](https://user-images.githubusercontent.com/25688193/29602649-f6be2f7a-881b-11e7-92d7-09e7a0a5ed20.png)

*    *    *

<a name="#EMアルゴリズム"></a>

## EMアルゴリズム

![twitter_ _ _11-6_170701](https://user-images.githubusercontent.com/25688193/29602472-4746aeb4-881b-11e7-96c2-9c55d6a5bcbd.png)
![twitter_ _ _11-7_170701](https://user-images.githubusercontent.com/25688193/29602478-4768b040-881b-11e7-83fb-aa882a5a42fe.png)
![twitter_ _ _11-7 _170701](https://user-images.githubusercontent.com/25688193/29602482-476dbc98-881b-11e7-8172-4aa6c562418a.png)
![twitter_ _ _11-8_170701](https://user-images.githubusercontent.com/25688193/29602483-476e0252-881b-11e7-8c87-e001e8253c45.png)
![twitter_ _ _11-9_170701](https://user-images.githubusercontent.com/25688193/29602480-476c2400-881b-11e7-95cc-16e8a166dc1f.png)
![twitter_ _ _11-10_170701](https://user-images.githubusercontent.com/25688193/29602481-476c61c2-881b-11e7-99c0-5b5f5f06633b.png)

*    *    *

