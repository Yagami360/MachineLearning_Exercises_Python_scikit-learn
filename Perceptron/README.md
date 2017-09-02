## パーセプトロン [Perceptron]（scikit-learn ライブラリ不使用）

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [パーセプトロンによる重みベクトルの更新と識別処理](#パーセプトロンによる重みベクトルの更新と識別処理)
    1. [](#)
1. [背景理論](#背景理論)
    1. [](#)


<a name="#使用するライブラリ"></a>

### 使用するライブラリ

</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : </br>
> https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#パーセプトロンによる重みベクトルの更新と識別処理"></a>

## パーセプトロンによる重みベクトルの更新と識別処理 : `main.py`

- Iris データセットを使用 : `pandas.read_csv()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 100 個 :</br> `X_features = df_Iris.iloc[0:100, [0,2]].values`
- サンプル数 100 個の内、品種 "setosa" が 50 個、"virginica" が 50 個。
- 教師データ `y_labels` は、サンプル数 100 個 : </br >`y_labels = df_Iris.iloc[0:100,4].values`
    - カテゴリーデータを -1 or 1 に変換 : </br>`y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1)`
- 正規化処理を実施しない。</br> 
- 自作クラス `Perceptron` でパーセプトロンの fitting 処理（重みベクトルの更新）を実施 :</br>
`Perceptron.fit( X_features, y_labels )`

</br>

>アヤメデータを単一パーセプトロン＆最急降下法で識別結果 </br>（重みの更新:Δw = η*(y_i-y^_i)）
![twitter_ _1_2_170718](https://user-images.githubusercontent.com/25688193/28357345-0fc51218-6ca6-11e7-859e-5e1d71bca1c2.png)

</br>

---

<a name="#背景理論"></a>

## 背景理論
