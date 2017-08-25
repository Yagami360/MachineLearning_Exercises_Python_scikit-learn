## アンサンブル学習 [Ensemble Learning]

コード実装中...

### 目次 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [ : `main1().py`](#)
    1. [ : `main2().py`](#)
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

### アンサンブル法と、単体での分類器での誤分類率の比較 : </br> `main1().py`

>最終的な識別結果を複数の分類器での多数決で決めるアンサンブル法（２項分布の累積に従う）と、単体での分類器での誤分類率の比較図。</br>
分類器の個数が奇数で、ランダムな結果（0.5）より識別性能が高い（＝図では 0.5 より小さい領域）場合、アンサンブルな手法のほうが、単体での分類器より、より誤識別が常に勝っていることが分かる。</br>
分類器の個数が偶数個の場合は、必ずしもこれが成り立つとは限らないことに注意（アンサンブル法では多数決で最終予想を決めているため。）

![ensemblelearning_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29705020-33fd8704-89b7-11e7-9760-5d04bca26af6.png)



- Iris データセットを使用
- トレーニングデータ 80% 、テストデータ 20%の割合で分割
- scikit -learn ライブラリ の `Pipeline` クラスを使用して、各機械学習プロセスを実施
  - パイプラインの１つ目の変換器は、正規化処理 : </br>`("scl", StandardScaler())`
  - パイプラインの２つ目の変換器は、PCA による次元削除（ 30 → 2 次元 ） : </br>`( "pca", PCA( n_components=2 ) )`
  - パイプラインの推定器は、ロジスティクス回帰 : </br>`( "clf", LogisticRegression( random_state=1 )`
- クロス・バディゲーション（k=10）で汎化性能を評価 : </br>`sklearn.model_selection.cross_val_score()` を使用

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

