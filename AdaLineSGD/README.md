## AdaLineSGD（Adaptive Liner ニューロン確率的勾配降下法）</br> [Adaptive Liner stomatic Gradient descent]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [AdaLineSGD によるアヤメデータの識別と、重みベクトルの更新処理](#AdaLineSGD によるアヤメデータの識別と、重みベクトルの更新処理)
    1. [](#)
1. [背景理論](#背景理論)
    1. [](#)


<a name="#使用するライブラリ"></a>

### 使用するライブラリ


<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : csv フォーマット </br>
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data


<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#AdaLineSGD によるアヤメデータの識別と、重みベクトルの更新処理"></a>

## AdaLineSGD によるアヤメデータの識別と、重みベクトルの更新処理 : `main.py`

- Iris データセットを使用
- 特徴行列 `X_features` は、特徴数 2 個（ Sepal Width, Petal length）× サンプル数 100 個 :</br> `X_features = df_Iris.iloc[0:100, [0,2]].values`
- サンプル数 100 個の内、品種 "setosa" が 50 個、"virginica" が 50 個。
- 教師データ `y_labels` は、サンプル数 100 個 : </br >`y_labels = df_Iris.iloc[0:100,4].values`
    - カテゴリーデータを -1 or 1 に変換 : </br>`y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1)`
- 正規化処理を実施する。</br> 
    - 正規化処理 : </br>
    `X_features_std = numpy.copy( X_features ) # ディープコピー`</br>
    `X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std() `</br>
    `X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()`
- 自作クラス `AdaLineSGD` を用いて、学習データでの fitting 処理（重みベクトルの更新処理）を行う。</br> `AdaLineSGD.fit( X_features, y_labels )`
- 自作クラス `AdaLineSGD` の `online_fit()` 関数を用いて、"擬似的な"ストリーミングデータで、"擬似的な"オンライン学習する。
```
    # ストリーミングデータ (5 ~ 10) でオンライン学習
    for smIndex in range(5,100):
        print(smIndex)
        ada2.online_fit( 
            X_train = X_features_std[0:smIndex, :], 
            y_train = y_labels[0:smIndex] 
        )
```

</br>

> アヤメデータをAdaLine＆確率的最急降下法（コスト関数）、及びオンライン学習で識別結果。</br>（重みの更新：Δw=η*( y_i - Φ(w^T*x_i) ), J=(1/2)*( y_i - Φ(w^T*x_i) )^2, i：ランダム）
![twitter_adaline_2-2_170719](https://user-images.githubusercontent.com/25688193/28357356-19940cb8-6ca6-11e7-80ba-50e0c968f6dc.png)

</br>

---

<a name="#背景理論"></a>

## 背景理論

<a name="#背景理論１"></a>

## 背景理論１

<a name="#背景理論２"></a>

## 背景理論２
