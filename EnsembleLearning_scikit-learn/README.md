## アンサンブルモデルとアンサンブル学習 [Ensemble Learning]

### 目次 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [多数決方式のアンサンブル法と、単体での分類器での誤分類率。及び多数決方式のアンサンブル法における分類器の個数に応じた比較](#main1.py)
    1. [多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ](#多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ)
        1. [アヤメデータでの検証結果](#アヤメデータでの検証結果)
        1. [同心円状データでの検証結果](#同心円状データでの検証結果)
        1. [半月状データでの検証結果](#同心円状データでの検証結果)
    1. [バギングの実行結果](#バギングの実行結果)
    1. [アダブーストの実行結果](#アダブーストの実行結果)
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

>>>> scikit-learn の推定器 estimator の基本クラス BaseEstimator を継承した自作クラスを作成していたのだが、コンストラクタ `__init()__` で指定した引数と同名のオブジェクトの属性を設定する必要ある模様。ドキュメントにそれらしき記載あり。
![image](https://user-images.githubusercontent.com/25688193/29766807-0c799194-8c1b-11e7-86c0-a63bed2e3233.png)

> その他ライブラリ
>> `math` :</br>
https://docs.python.org/2/library/math.html
>> `scipy.misc.comb` :</br>
https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.comb.html


<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データ : `sklearn.datasets.load_iris()`</br>
> 同心円状のデータセット : `sklearn.datasets.make_circles()` </br>
> 半月状のデータセット : `sklearn.datasets.make_moons()` </br>

---

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#main1.py"></a>

### 多数決方式のアンサンブル法と、単体での分類器での誤分類率。及び多数決方式のアンサンブル法における分類器の個数に応じた比較 : `main1.py`

>多数決方式のアンサンブル法（最終的な識別結果を複数の分類器での多数決で決め、２項分布の累積に従う）と、単体での分類器での誤分類率の比較図、及び多数決方式のアンサンブル法における分類器の個数に応じた比較図。</br>
分類器の個数が奇数で、ランダムな結果（0.5）より識別性能が高い（＝図では 0.5 より小さい領域）場合、多数決方式でのアンサンブルな手法のほうが、単体での分類器より、常に誤識別が低い（識別性能が高い）ことが分かる。</br>
尚、分類器の個数が偶数個の場合は、必ずしもこれが成り立つとは限らないことに注意（この多数決方式のアンサンブル法では多数決により最終予想を決めるため。）

![ensemblelearning_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29705020-33fd8704-89b7-11e7-9760-5d04bca26af6.png)

</br>

<a name="#多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ"></a>

### 多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ : `main2.py`

アンサンブルモデルの強みである多様性を確認するために、単純な重み付け多数決方式でのアンサンブルモデルでの検証をしてみた。（より実用的なのは、同じアンサンブル学習で、Random Forest や AdaBoost, XGBoost で）

- アンサンブル法による分類器の自作クラス `EnsembleModelClassifier` を使用
- この自作クラス `EnsembleModelClassifier` に scikit -learn ライブラリ の `Pipeline` クラスを設定 : </br> 
`EnsembleModelClassifier( classifiers  = [ pipe1, pipe2, pipe3 ], class_lebels = [...] )`</br>
`pipe1 = Pipeline( steps =[ ( "sc", StandardScaler() ), ( "clf", clf1 ) ] )`
    > 以下、パイプラインの構成例    
    >> １つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`</br>
    >> １つ目のパイプラインの推定器は、ロジスティクス回帰 : </br>`( "clf", LogisticRegression( penalty = 'l2', C = 0.001, random_state = 0 )`</br>

    >> ２つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`</br>
    >> ２つ目のパイプラインの推定器は、決定木 : </br>`( "clf", DecisionTreeClassifier( max_depth = 3,  criterion = 'entropy', random_state = 0 )`
    
    >>３つ目のパイプラインの１つ目の変換器は、正規化処理 : </br>`("sc", StandardScaler())`</br>
    >> ３つ目のパイプラインの推定器は、k-NN 法 : </br>`( "clf", KNeighborsClassifier( n_neighbors = 3, p = 2, metric = 'minkowski' )`</br>
- クロス・バディゲーション k-fold CV (k=10) で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score( cv=10 )`


<a name="アヤメデータでの検証結果"></a>

#### Iris データセットでの検証結果

- Iris データ : `datasets.load_iris()`
- 特徴行列（特徴量2個×50個のサンプル） : `iris.data[ 50:, [1, 2] ]`
- 教師データ（50個のサンプル） : `iris.target[50:]`
- トレーニングデータ 50% 、テストデータ 50%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.5, random_state = 1 )`
- パイプラインの変換器で正規化処理実施 :</br>`("sc", StandardScaler())`
- クロス・バディゲーション k-fold CV (k=10) で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score( cv=10 )`

>各種スコア値の表 by k-fold CV (cv=10) :（チューニング前）

|Model (classifiers)|Accuracy</br>[train data]|Accuracy</br>[test data]|AUC</br>[train data]|AUC</br>[test data]|
|---|---|---|---|---|
|Logistic Regression </br> `penalty = 'l2', C = 0.001`|0.84 (+/- 0.23)|0.83 (+/- 0.18)|0.92 (+/- 0.20)|0.95 (+/- 0.11)|
|Decision Tree </br> `criterion = 'entropy', max_depth = 3`|0.92 (+/- 0.13)|0.81 (+/- 0.16)|0.92 (+/- 0.15)|0.85 (+/- 0.14)|
|k-NN </br> `n_neighbors = 3, metric = 'minkowski'`|0.92 (+/- 0.13)|0.83 (+/- 0.14)|0.93 (+/- 0.12)|0.89 (+/- 0.11)|
|SVM</br> ` kernel = 'rbf', C = 10.0, gamma = 0.50 `|0.94 (+/- 0.09)|0.88 (+/- 0.10)|1.00 (+/- 0.00)|0.94 (+/- 0.08)|
|Ensemble Model 1</br> [LogisticRegression, DecisionTree, k-NN]|0.92 (+/- 0.13)|0.83 (+/- 0.17)|0.93 (+/- 0.20)|0.94 (+/- 0.09)|
|Ensemble Model 2</br> [LogisticRegression, DecisionTree, SVM]|0.92 (+/- 0.13)|0.86 (+/- 0.13)|1.00 (+/- 0.00)|0.91 (+/- 0.09)|

</br>

> 識別結果＆識別境界（チューニング前）
>> チューニング前の適当なハイパーパラメータでの各弱識別器＆これらのアンサンブルモデルでの識別結果＆識別境界の図。１枚目の図は、弱識別器として｛ロジスティクス回帰、決定木、k-NN法｝からなるアンサンブルモデル。２枚目の図は、弱識別器として｛ロジスティクス回帰、決定木、SVM｝からなるアンサンブルモデル。図より、アンサンブルモデルでは、これを構成する個々の弱識別器の識別境界を混ぜ合わせた形状になっていることが分かる。
![ensemblelearning_scikit-learn_2-1](https://user-images.githubusercontent.com/25688193/29752697-1f7f2f1a-8b9e-11e7-9972-308010923fd3.png)
![ensemblelearning_scikit-learn_2-2](https://user-images.githubusercontent.com/25688193/29752701-21eadbb4-8b9e-11e7-8c7a-ff738e6d9f5f.png)

> 学習曲線（チューニング前）
![ensemblelearning_scikit-learn_3-1](https://user-images.githubusercontent.com/25688193/29752710-3388f900-8b9e-11e7-9d3c-d1f89aa11031.png)
![ensemblelearning_scikit-learn_3-2](https://user-images.githubusercontent.com/25688193/29752711-360c57da-8b9e-11e7-8427-a8b5604812cc.png)

> ROC 曲線（チューニング前）
![ensemblelearning_scikit-learn_4-1](https://user-images.githubusercontent.com/25688193/29752717-43e7f7ba-8b9e-11e7-868c-52a35e7831b3.png)
![ensemblelearning_scikit-learn_4-1](https://user-images.githubusercontent.com/25688193/29752718-44fe507c-8b9e-11e7-947a-bc2ee3099d47.png)
![ensemblelearning_scikit-learn_4-2](https://user-images.githubusercontent.com/25688193/29752724-504f6c54-8b9e-11e7-9b27-b2cf0703a01e.png)
![ensemblelearning_scikit-learn_4-2](https://user-images.githubusercontent.com/25688193/29752725-5242df14-8b9e-11e7-8544-a8cdb0229eff.png)

</br>

> グリッドサーチによる各弱識別器のチューニング</br>

>> `EnsembleModelClassifer.get_params()` で得られるパラメータのディクショナリ構造。</br>
>> グリッドサーチ `sklearn.model_selection.GridSearchCV()` での使用を想定した構造になっている。</br>
>> 詳細には、ディクショナリの Key の内、モデルのハイパーパラメータとなっいるディクショナリを設定する。</br>
>>> 例 : `sklearn.model_selection.GridSearchCV()` に渡す引数 `parames` の設定</br>
>>> `params ={ pipeline-1__clf__C":[0.001, 0.1, 1, 10, 100.0],` </br>
>>> `"pipeline-2__clf__max_depth": [1, 2, 3, 4, 5] }`

|Key|values|
|---|---|
|'pipeline-1':|Pipeline( </br>steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ('clf', LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False))]), |
|'pipeline-2':|Pipeline(</br>steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ('clf', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=0, splitter='best'))]</br>), |
|'pipeline-3':|Pipeline(</br>steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma=0.5, kernel='rbf', max_iter=-1, probability=True, random_state=0, shrinking=True, tol=0.001, verbose=False))]),|
|'pipeline-1__steps':| [('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ('clf', LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False))], |
|'pipeline-1__sc':|StandardScaler(copy=True, with_mean=True, with_std=True),|
|'pipeline-1__clf':|LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False), |
|'pipeline-1__sc__copy':|True, |
|'pipeline-1__sc__with_mean':| True,|
|'pipeline-1__sc__with_std':|True, |
|'pipeline-1__clf__C': |0.001, |
|'pipeline-1__clf__class_weight':|None, |
|'pipeline-1__clf__dual': |False, |
|'pipeline-1__clf__fit_intercept':|True,|
|'pipeline-1__clf__intercept_scaling':| 1, |
|'pipeline-1__clf__max_iter': |100, |
|'pipeline-1__clf__multi_class':|'ovr', |
|'pipeline-1__clf__n_jobs':| 1,|
|'pipeline-1__clf__penalty': |'l2', |
|'pipeline-1__clf__random_state':| 0,| 
|'pipeline-1__clf__solver': |'liblinear', |
|'pipeline-1__clf__tol':| 0.0001,|
|'pipeline-1__clf__verbose': |0, |
|'pipeline-1__clf__warm_start': |False, |
|'pipeline-2__steps': |[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)),</br> ('clf', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=0, splitter='best'))], |
|'pipeline-2__sc': |StandardScaler(copy=True, with_mean=True, with_std=True), |
|'pipeline-2__clf': |DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=0, splitter='best'), |
|'pipeline-2__sc__copy': |True, |
|'pipeline-2__sc__with_mean': |True,|
|'pipeline-2__sc__with_std': |True, |
|'pipeline-2__clf__class_weight': |None, |
|'pipeline-2__clf__criterion': |'entropy',| 
|'pipeline-2__clf__max_depth': |3,|
|'pipeline-2__clf__max_features': |None, |
|'pipeline-2__clf__max_leaf_nodes': |None, |
|'pipeline-2__clf__min_impurity_split':| 1e-07, |
|'pipeline-2__clf__min_samples_leaf':| 1, |
|'pipeline-2__clf__min_samples_split':| 2, |
|'pipeline-2__clf__min_weight_fraction_leaf':| 0.0, |
|'pipeline-2__clf__presort': |False, |
|'pipeline-2__clf__random_state':| 0, |
|'pipeline-2__clf__splitter': |'best', |
|'pipeline-3__steps':|[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), </br>('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma=0.5, kernel='rbf', max_iter=-1, probability=True, random_state=0, shrinking=True, tol=0.001, verbose=False))], |
|'pipeline-3__sc': | StandardScaler(copy=True, with_mean=True, with_std=True), |
|'pipeline-3__clf': | SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma=0.5, kernel='rbf', max_iter=-1, probability=True, random_state=0, shrinking=True, tol=0.001, verbose=False), |  
|'pipeline-3__sc__copy': |True, |
|'pipeline-3__sc__with_mean': |True, |
|'pipeline-3__sc__with_std': |True, |
|'pipeline-3__clf__C': |10.0, |
|'pipeline-3__clf__cache_size': |200, |
|'pipeline-3__clf__class_weight': |None,| 
|'pipeline-3__clf__coef0': |0.0, |
|'pipeline-3__clf__decision_function_shape':| None,| 
|'pipeline-3__clf__degree':| 3,| 
|'pipeline-3__clf__gamma':| 0.5, |
|'pipeline-3__clf__kernel': |'rbf', |
|'pipeline-3__clf__max_iter':| -1,| 
|'pipeline-3__clf__probability':| True, |
|'pipeline-3__clf__random_state':| 0, |
|'pipeline-3__clf__shrinking':| True, |
|'pipeline-3__clf__tol':| 0.001, |
|'pipeline-3__clf__verbose':| False}|

</br>

> グリッドサーチ結果
>> Ensemble Mode2 ( LogisticRegression, DecisionTree, SVM)

|ハイパーパラメータ|Key|Best parameters|Accuracy|
|---|---|---|---|
|LogisticRegression の C 値|pipeline-1__clf__C|...|...|
|DecisionTree の depth 値|pipeline-2__clf__max_depth|...|...|...|
|SVM の C 値, gamma 値|pipeline-3__clf__C, </br >pipeline-3__clf__gamma|{'pipeline-3__clf__C': 0.1, '</br>pipeline-3__clf__gamma': 10}|1.00|
|...|...|...|...|

</br>
</br>

<a name="同心円状データセットでの検証結果"></a>

#### 同心円状データセットでの検証結果

- 検証用データとして、同心円状データセットを使用 : </br> 
`sklearn.datasets.make_circles( n_samples = 1000, random_state = 123, noize = 0.1, factor = 0.2 )`
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 :</br> `sklearn.model_selection.train_test_split()`
- パイプライン経由で正規化処理実施 :</br>
- クロス・バディゲーション k-fold CV (k=10) で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score( cv=10 )`

> 各種スコア値の表 by k-fold CV : `cross_val_score( cv = 10 )`</br>
...

> 各モデルでの識別境界
![ensemblelearning_scikit-learn_2-3](https://user-images.githubusercontent.com/25688193/29753047-34f346e6-8ba4-11e7-8eab-1b9fa91b2141.png)

> 学習曲線
![ensemblelearning_scikit-learn_3-3](https://user-images.githubusercontent.com/25688193/29753048-36aaec78-8ba4-11e7-80d8-11e9f1027c81.png)

> ROC 曲線、AUC 値
![ensemblelearning_scikit-learn_4-3](https://user-images.githubusercontent.com/25688193/29753052-3a43c4ae-8ba4-11e7-93bc-d5ecd6119c6a.png)

</br>

<a name="半月状データセットでの検証結果"></a>

#### 半月形のデータセットでの検証結果

- 検証用データとして、半月状データセットを使用 : </br> 
`sklearn.datasets.make_moons( n_samples = 1000, random_state = 123 )`
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 :</br> `sklearn.model_selection.train_test_split()`
- パイプライン経由で正規化処理実施 :</br>
- クロス・バディゲーション k-fold CV (k=10) で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score( cv=10 )`

> 各モデルでの識別境界
![ensemblelearning_scikit-learn_2-4](https://user-images.githubusercontent.com/25688193/29753081-ed99328c-8ba4-11e7-9818-d47ded5ae6e0.png)

> 学習曲線
![ensemblelearning_scikit-learn_3-4](https://user-images.githubusercontent.com/25688193/29753083-f1664486-8ba4-11e7-80fb-b8f22a032629.png)

> ROC 曲線、AUC 値
![ensemblelearning_scikit-learn_4-4](https://user-images.githubusercontent.com/25688193/29753084-f31ef610-8ba4-11e7-8dbb-5b547718c77d.png)

</br>

<a name="#バギングの実行結果"></a>

### バギング の実行結果 : `main3.py`

> コード実装中...

</br>

<a name="#アダブーストの実行結果"></a>

### アダブーストの実行結果 : `main4.py`

> コード実装中...

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

