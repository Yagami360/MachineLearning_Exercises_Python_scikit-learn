## scikit-learn パイプライン（Pipeline クラス）による機械学習処理フローの効率化、及び汎化性能の各種評価方法


### 使用する scikit-learn ライブラリ：

> パイプライン
>> `sklearn.pipeline.Pipeline` :
  http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

> クロスバディゲーション
>> `sklearn.model_selection.StratifiedKFold` : http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
> `sklearn.model_selection.cross_val_score` : 
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

> 学習曲線、検証曲線
>> `sklearn.model_selection.learning_curve` : http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html


### 使用するデータセット

> Brest Cancer Wisconsin データセット（csvフォーマット）: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
>>悪性腫瘍細胞と良性腫瘍細胞の 569 個のサンプルが含まれている。1 列目は固有の ID 、2 列目は悪性 [malignant] or 良性 [belign] を表すラベル、3 ~ 32 列目には、細胞核のデジタル画像から算出された 30 個の実数値の特徴量が含まれれいる。

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

### クロスバディゲーションを用いた、モデルの汎化能力の評価 : `main1().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各プロセスを実施
  - パイプラインの１つ目の推定器は、スケーリング処理 : `("scl", StandardScaler())`
  - パイプラインの２つ目の推定器は、PCA による次元削除（ 30 → 2 次元 ） : `( "pca", PCA( n_components=2 ) )`
  - パイプラインの予想器は、ロジスティクス回帰 : `( "clf", LogisticRegression( random_state=1 )`
- クロス・バディゲーションで汎化性能を評価 : `sklearn.model_selection.cross_val_score()` を使用

> クロス・バディゲーションでの汎化性能の検証結果

CV accuracy scores :

|k|1|2|3|4|5|6|7|8|9|10|
|---|---|---|---|---|---|---|---|---|---|---|
|正解率|0.978|0.978|0.957|0.978|0.891|0.956|0.956|0.956|1.|0.978|

CV accuracy :

|平均値|分散値|
|---|---|
|0.963|+/- 0.028|

### 学習曲線, 検証曲線よるモデルの汎化性能の評価 : `main2().py`

- Brest Cancer Wisconsin データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各プロセスを実施
  - パイプラインの１つ目の推定器は、スケーリング処理 : `("scl", StandardScaler())`
  - パイプラインの予想器は、ロジスティクス回帰 : `( "clf", LogisticRegression( random_state=1 )`
- 学習曲線で汎化性能（バイアス・バリアントトレードオフ）を評価 : 
- 検証曲線で汎化性能（過学習、学習不足）を評価 : 

> 学習曲線

コード実施中...

> 検証曲線

コード実施中...

### グリッドサーチによるモデルのハイパーパラメータのチューニング : `main3().py`

コード実施中...

### ROC 曲線よるモデルの汎化性能の評価 : `main4().py`

コード実施中...

---

## Theory

### モデル選択とチューニングパラメータ（ハイパーパラメータ）

「モデル選択」という用語は、チューニングパラメータの”最適な”値を選択する分類問題を指す。

![twitter_ _3-1_160924](https://user-images.githubusercontent.com/25688193/29446078-4d1ba9a6-8425-11e7-8011-243be82de1db.png)
![twitter_ _3-2_160924](https://user-images.githubusercontent.com/25688193/29446080-4d20feec-8425-11e7-9f74-8395a4521459.png)
![twitter_ _3-3_160924](https://user-images.githubusercontent.com/25688193/29446079-4d1dd51e-8425-11e7-9cee-372aec8bca8d.png)

*    *    *

### トレーニングデータ、テストデータへのデータの分割

![twitter_ 4-1_160918](https://user-images.githubusercontent.com/25688193/29446101-65b4cfb0-8425-11e7-8b7b-b481ad353160.png)


### ホールドアウト法による汎化能力の検証

![twitter_ 5-1_160918](https://user-images.githubusercontent.com/25688193/29446118-7719d1d8-8425-11e7-9282-0de039e5ac43.png)

### クロスバディゲーション法による汎化能力の検証

![twitter_ 5-2_160919](https://user-images.githubusercontent.com/25688193/29446124-7bb14b86-8425-11e7-901a-8817811bea17.png)

*    *    *

### 学習曲線、検証曲線による汎化能力の検証とバイアス・バリアントトレードオフ

*    *    *

### グリッドサーチによるハイパーパラメータのチューニング

*    *    *

### ROC 曲線による汎化能力の検証

![twitter_ 10-1_161005](https://user-images.githubusercontent.com/25688193/29446156-a1324964-8425-11e7-9555-da0dc49f132c.png)
![twitter_ 10-2_161005](https://user-images.githubusercontent.com/25688193/29446159-a13575da-8425-11e7-97bc-277d49f09136.png)
![twitter_ 10-3_161005](https://user-images.githubusercontent.com/25688193/29446160-a136d20e-8425-11e7-8451-cb0440cc4d5a.png)
![twitter_ 11-1_161005](https://user-images.githubusercontent.com/25688193/29446161-a1383a9a-8425-11e7-86b0-c23fe22e0039.png)
![twitter_ 11-3_161005](https://user-images.githubusercontent.com/25688193/29446158-a134fe98-8425-11e7-9cda-37a0585357e8.png)
![twitter_ 12-1_161004](https://user-images.githubusercontent.com/25688193/29446157-a1336a06-8425-11e7-9f16-d66eb9bb9927.png)

---

![twitter_ 13-1_161227](https://user-images.githubusercontent.com/25688193/29446164-a15b0ab6-8425-11e7-97e0-dc557c10ce99.png)
![twitter_ 13-2_161227](https://user-images.githubusercontent.com/25688193/29446162-a15952c0-8425-11e7-99c4-13c679b7a3d3.png)
![twitter_ 13-3_161229](https://user-images.githubusercontent.com/25688193/29446163-a15b0912-8425-11e7-972e-0a8ddc9c2097.png)
![twitter_ 13-5_170101](https://user-images.githubusercontent.com/25688193/29446165-a15b9f76-8425-11e7-9e1c-aa9e70f7ddc1.png)
![twitter_ 13-8_170102](https://user-images.githubusercontent.com/25688193/29446167-a167e70e-8425-11e7-81db-502156d96cfd.png)
![twitter_ 13-9_170102](https://user-images.githubusercontent.com/25688193/29446166-a164b598-8425-11e7-9de7-ec1d8f7e9fc1.png)
![twitter_ 13-10_170102](https://user-images.githubusercontent.com/25688193/29446168-a180224c-8425-11e7-8d74-ac831ac73c26.png)

*    *    *
