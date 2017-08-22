## scikit-learn パイプライン（`Pipeline` クラス）による機械学習処理フローの効率化、</br> 及び、モデルの汎化性能の各種評価方法（２クラスの識別問題を対象）


### 使用する scikit-learn ライブラリ：

> パイプライン
>> `sklearn.pipeline.Pipeline` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

> クロスバディゲーション
>> `sklearn.model_selection.StratifiedKFold` :</br> http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
>> `sklearn.model_selection.cross_val_score` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

> 学習曲線、検証曲線
>> `sklearn.model_selection.learning_curve` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
>> `sklearn.model_selection.validation_curve` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html

> グリッドサーチ
>> `sklearn.model_selection.GridSearchCV` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
</br> ＜補足＞ ヒートマップ `seaborn.heatma` : </br> 
https://seaborn.pydata.org/generated/seaborn.heatmap.html </br> http://pythondatascience.plavox.info/seaborn/heatmap

> 混同行列、適合率、再現率、F1スコア
>> `sklearn.metrics.confusion_matrix` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
>> `sklearn.metrics.precision_score` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
>> `sklearn.metrics.recall_score` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
>> `sklearn.metrics.f1_score` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

> ROC 曲線
>> `sklearn.metrics.roc_curve` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
>> `sklearn.metrics.auc` : </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html


### 使用するデータセット

> Brest Cancer Wisconsin データセット（csvフォーマット）:</br>
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
>>悪性腫瘍細胞と良性腫瘍細胞の 569 個のサンプルが含まれている。</br> 1 列目は固有の ID 、2 列目は悪性 [malignant] or 良性 [belign] を表すラベル、3 ~ 32 列目には、細胞核のデジタル画像から算出された 30 個の実数値の特徴量が含まれれいる。

|行番号|ID|悪性（M）/良性（B）|1|2|3|4|5|6|7|8|...|22|23|24|25|26|27|28|29|30|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|842302  |M  |17.990  |10.38  |122.80  |1001.0  |0.11840  |0.27760  |0.300100|0.147100|...|25.380|17.33|  184.60  |2019.0  |0.16220|  0.66560| 0.71190|  0.26540|  0.4601|  0.11890 |
|2|842517  |M  |20.570  |17.77  |132.90  |1326.0  |0.08474  |0.07864  |0.086900|0.147100|...|25.380|  17.33  |184.60  |2019.0  |0.16220  |0.66560|0.24160  |0.18600  |0.2750  |0.08902|
|3|84300903  |M  |19.690  |21.25  |130.00  |1203.0  |0.10960  |0.15990  |0.197400|0.127900   |...|23.570  |25.53  |152.50  |1709.0  |0.14440  |0.42450|0.45040  |0.24300  |0.3613  |0.08758|
|...|
|568|927241  |M  |20.600  |29.33  |140.10  |1265.0  |0.11780  |0.27700  |0.351400|0.152000   |...|25.740  |39.42  |184.60  |1821.0  |0.16500  |0.86810|0.93870  |0.26500  |0.4087  |0.12400|
|569|92751  |B   |7.760  |24.54   |47.92   |181.0  |0.05263  |0.04362  |0.000000|0.000000|...|9.456  |30.37   |59.16   |268.6  |0.08996  |0.06444|0.00000  |0.00000  |0.2871  |0.07039|

---

## コードの実行結果

### クロスバディゲーションを用いた、モデルの汎化能力の評価 : </br>`main1().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br>`("scl", StandardScaler())`
  - パイプラインの２つ目の変換器は、PCA による次元削除（ 30 → 2 次元 ） : </br>`( "pca", PCA( n_components=2 ) )`
  - パイプラインの推定器は、ロジスティクス回帰 : </br>`( "clf", LogisticRegression( random_state=1 )`
- クロス・バディゲーション（k=10）で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score()` を使用

> クロス・バディゲーションでの汎化性能の検証結果
>> CV accuracy scores :

|k|1|2|3|4|5|6|7|8|9|10|
|---|---|---|---|---|---|---|---|---|---|---|
|正解率|0.978|0.978|0.957|0.978|0.891|0.956|0.956|0.956|1.|0.978|

>> CV accuracy :

|平均値|分散値|
|---|---|
|0.963|+/- 0.028|

### 学習曲線, 検証曲線よるモデルの汎化性能の評価 : </br>`main2().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br> `("scl", StandardScaler())`
  - パイプラインの推定器は、ロジスティクス回帰（L2正則化） : </br> `( "clf", LogisticRegressionLogisticRegression(penalty='l2', random_state=0)`
  - このロジスティクス回帰は、交差エントロピー関数（評価関数）を L2 正則化する。（過学習対策）
- クロス・バディゲーション（k=10）で汎化性能を評価 : 
- 学習曲線で汎化性能（バイアス・バリアントトレードオフ関係）を評価 :</br> `learning_curve()`
- 検証曲線で汎化性能（過学習、学習不足）を評価 : </br>`validation_curve()`

> 学習曲線 [Learning Curve]
>>図より、トレーニングサンプル数が、325 個程度を超えたあたりから僅かな過学習が発生しているのが見て取れるが、全体的にバイアス・バリアントトレードオフがちょうどいいバランスになっていることが分かる。

![machinelearningpipeline_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29451212-c7290d4e-843c-11e7-9103-d33cc5aa1b6e.png)

> 検証曲線 [Validation Curve]
>> 横軸は、パイプラインの推定器に使用したこのロジスティクス回帰の交差エントロピー関数（評価関数）を L2 正則化するための、逆正則化パラメータ C の値（log スケール）。</br> C 値を小さくして、正則化の強さを上げる（図中の右から左方向）ほど、過学習傾向が下がっていることが見て取れる。つまり、L２正則化による過学習対策がうまく出来ていることが分かる。

![machinelearningpipeline_scikit-learn_2](https://user-images.githubusercontent.com/25688193/29456506-ec919114-8450-11e7-99f7-b20f0c230a57.png)


### グリッドサーチに [grid search] よるモデルのハイパーパラメータのチューニング : </br>`main3().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br> `("scl", StandardScaler())`
  - パイプラインの推定器は、サポートベクターマシン（C-SVM） : </br> `( "clf", 'clf', SVC( random_state = 1 )`
- クロス・バディゲーション（k=10）で汎化性能を評価 : 
- scikit -learn ライブラリ の `GridSearchCV()` 関数を使用して、ハイパーパラメータのグリッドサーチを実行

> グリッドサーチが対象とするパラメータのヒートマップ : 作図に `seaborn.heatmap()` を使用
>> 以下のヒートマップ図より、推定器として、RBF-kernel SVM を使用した場合、最も正解率が高くなるパラメータ（ハイパーパラメータ）は、C = 10, gamma = 0.01 (Accuracy = 0.980) となることが分かる。

![machinelearningpipeline_scikit-learn_3](https://user-images.githubusercontent.com/25688193/29512943-bc59e370-869e-11e7-8c4c-e4bac940d390.png)


> `sklearn.model_selection.GridSearchCV` を使用したグリッドサーチの結果

|グリッドサーチの結果|values|
|---|---|
|最もよいスコアを出したモデルの正解率 [平均値 +/- 分散値] </br> `sklearn.model_selection.GridSearchCV.best_score_`|0.980 (+/-0.021)|
|最もよいスコアを出したモデルのパラメータ </br> `sklearn.model_selection.GridSearchCV.best_params_`| RBF-kernel SVM の C 値 : 10.0, ガンマ値 : 0.01 </br> {'clf__C': 10.0, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}|
|最もよいスコアを出したモデルでのテストデータでの正解率 [平均値 +/- 分散値] </br> `sklearn.model_selection.GridSearchCV.best_estimator_`| 0.982 (+/- 0.000)|
|最もよいスコアを出したモデルでの</br>ハイパーパラメータの全組み合わせでの正解率の平均値 +/- 分散値</br>`sklearn.model_selection.GridSearchCV.grid_scores_`||

### 混同行列と ROC 曲線よるモデルの汎化性能の評価 : </br> `main4().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br> `("scl", StandardScaler())`
  - パイプラインの推定器は、サポートベクターマシン（C-SVM） : </br> `( "clf", 'clf', SVC( random_state = 1 )`
- クロス・バディゲーション（k=10）で汎化性能を評価 : 
- 混同行列 [confusion matrix] のヒートマップを作図し、性能評価 : </br> `sklearn.metrics.confusion_matrix`
- 適合率、再現率、F1 スコアで性能評価 : </br> `sklearn.metrics.precision_score` , `precision_score.recall_score` , `sklearn.metrics.f1_score` 
- ROC 曲線を作図し、性能評価 : </br> `sklearn.metrics.roc_curve` , `sklearn.metrics.auc`

> 混同行列 [confusion matrix] のヒートマップ
>> `sklearn.metrics.confusion_matrix` 関数で作成した混同行列を元に作図したヒートマップ図。</br>クラス０（悪性）を P（陽性クラス）、クラス１（陰性）を N（陰性クラス）とする。</br> 
このモデル（RBF-kernel SVM）は、</br> 
・ 真陰性 : TN [true negative]：（左上）悪性クラス（P）に属しているデータ 66 個を正しく識別 </br> 
・ 真陽性 : TP [true positive]：（右下）陽性クラス（N）に属しているデータ 46 個を正しく識別 </br> 
・ 偽陽性 : FP [false positive]（右上）悪性クラス（P）に属しているデータ 1 個を誤分類 </br> 
・ 偽陰性 : FN [false negative] : （左下）陽性クラス（N）に属しているデータ 1 個を誤分類 </br> 
尚、このデータの場合は、悪性であるものを陽性と誤分類するのが一番大きなリスクを伴うことに注意。</br>（つまり、単純に誤サンプル数では比較できない）

![machinelearningpipeline_scikit-learn_4](https://user-images.githubusercontent.com/25688193/29524147-fdc49eb0-86c9-11e7-97c6-e767c21f27d3.png)


> 適合率、再現率、F1 スコア

|各種性能指標|values|
|---|---|
|適合率 [Precision] <PRE = TP/(TP+FP)> </br> `sklearn.metrics.precision_score()`|0.979|
|再現率 [Recall] <REC = TP/(TP+FN)> </br> `sklearn.metrics.recall_score()`|0.979|
|F1 スコア <2×PRE×( REC/(PRE+REC) )> </br> `sklearn.metrics.f1_score()`|0.979|



> ROC 曲線 [Receiver Operator Characteristic Curve]
>>コード実装中...

---

## Theory

### モデル選択とチューニングパラメータ（ハイパーパラメータ）

> 広義な意味でのモデル選択

![twitter_ 20-1_170108](https://user-images.githubusercontent.com/25688193/29448075-e4d5dda2-842f-11e7-931c-7e861643466b.png)

![twitter_ _ _10-1_170625](https://user-images.githubusercontent.com/25688193/29447595-98f797a6-842d-11e7-9eaf-c228479483d7.png)
![twitter_ _ _10-2_170625](https://user-images.githubusercontent.com/25688193/29447597-9919f56c-842d-11e7-8251-e9c6e846c292.png)

> より、狭義な意味でのモデル選択
>>「モデル選択」という用語は、チューニングパラメータの”最適な”値を選択する分類問題を指す。

![twitter_ _3-1_160924](https://user-images.githubusercontent.com/25688193/29446078-4d1ba9a6-8425-11e7-8011-243be82de1db.png)

*    *    *

### トレーニングデータ、テストデータへのデータの分割

![twitter_ 4-1_160918](https://user-images.githubusercontent.com/25688193/29446101-65b4cfb0-8425-11e7-8b7b-b481ad353160.png)


### ホールドアウト法による汎化能力の検証

![twitter_ 5-1_160918](https://user-images.githubusercontent.com/25688193/29446118-7719d1d8-8425-11e7-9282-0de039e5ac43.png)

### クロスバディゲーション法による汎化能力の検証

![twitter_ 5-2_160919](https://user-images.githubusercontent.com/25688193/29446124-7bb14b86-8425-11e7-901a-8817811bea17.png)

*    *    *

### 学習曲線、検証曲線による汎化能力の検証とバイアス・バリアントトレードオフ

![twitter_ _3-2_160924](https://user-images.githubusercontent.com/25688193/29446080-4d20feec-8425-11e7-9f74-8395a4521459.png)
![twitter_ _3-3_160924](https://user-images.githubusercontent.com/25688193/29446079-4d1dd51e-8425-11e7-9cee-372aec8bca8d.png)

*    *    *

### グリッドサーチによるハイパーパラメータのチューニング

*    *    *

### ROC 曲線による汎化能力の検証

![twitter_ 13-1_161227](https://user-images.githubusercontent.com/25688193/29446164-a15b0ab6-8425-11e7-97e0-dc557c10ce99.png)
![twitter_ 13-2_161227](https://user-images.githubusercontent.com/25688193/29446162-a15952c0-8425-11e7-99c4-13c679b7a3d3.png)
![twitter_ 13-3_161229](https://user-images.githubusercontent.com/25688193/29446163-a15b0912-8425-11e7-972e-0a8ddc9c2097.png)
![twitter_ 13-5_170101](https://user-images.githubusercontent.com/25688193/29446165-a15b9f76-8425-11e7-9e1c-aa9e70f7ddc1.png)
![twitter_ 13-8_170102](https://user-images.githubusercontent.com/25688193/29446167-a167e70e-8425-11e7-81db-502156d96cfd.png)
![twitter_ 13-9_170102](https://user-images.githubusercontent.com/25688193/29446166-a164b598-8425-11e7-9de7-ec1d8f7e9fc1.png)
![twitter_ 13-10_170102](https://user-images.githubusercontent.com/25688193/29446168-a180224c-8425-11e7-8d74-ac831ac73c26.png)

*    *    *
