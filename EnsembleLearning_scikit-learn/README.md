## アンサンブル学習 [Ensemble Learning]

コード実施中...

### 目次 [Contents]

1. [使用する scikit-learn ライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [ : `main1().py`](#)
    1. [ : `main2().py`](#)
1. [背景理論](#背景理論)
    1. [](#)
    1. [](#)

<a name="#使用するライブラリ"></a>

### 使用する scikit-learn ライブラリ：

> パイプライン
>> `sklearn.pipeline.Pipeline` :</br>
http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html


<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データ:</br>
>>

|行番号|1|2|3|4|5|
|---|---|---|---|---|---|
|0||||||
|1||||||

---

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#"></a>

###  : </br> `main1().py`

- Iris データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br>`("scl", StandardScaler())`
  - パイプラインの２つ目の変換器は、PCA による次元削除（ 30 → 2 次元 ） : </br>`( "pca", PCA( n_components=2 ) )`
  - パイプラインの推定器は、ロジスティクス回帰 : </br>`( "clf", LogisticRegression( random_state=1 )`
- クロス・バディゲーション（k=10）で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score()` を使用

 
---

<a name="#背景理論"></a>

## 背景理論

<a name="#"></a>

### 


*    *    *

<a name="#"></a>

### 