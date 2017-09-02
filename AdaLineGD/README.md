## AdaLineGD（Adaptive Liner ニューロン勾配降下法） [Adaptive Liner Gradient descent]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [AdaLine によるアヤメデータの識別](#実行結果１)
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

<a name="#AdaLine によるアヤメデータの識別"></a>

## AdaLine によるアヤメデータの識別 : `main.py`

- Iris データセットを使用
- 特徴行列 `X_features` は、特徴数 2 個（ Sepal Width, Petal length）× サンプル数 100 個 :</br> `X_features = df_Iris.iloc[0:100, [0,2]].values`
- サンプル数 100 個の内、品種 "setosa" が 50 個、"virginica" が 50 個。
- 教師データ `y_labels` は、サンプル数 100 個 : </br >`y_labels = df_Iris.iloc[0:100,4].values`
    - カテゴリーデータを -1 or 1 に変換 : </br>`y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1)`
- 正規化処理を実施しないケースと実施したケースの双方で検証する。</br> 
    - 正規化処理 : </br>
    `X_features_std = numpy.copy( X_features ) # ディープコピー`</br>
    `X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std() `</br>
    `X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()`

> アヤメデータをAdaLine＆最急降下法（コスト関数）でのバッチ学習で識別結果。</br>（重みの更新:Δw=η*∑( y-Φ(w^T*x) ) (j=1,2,...,m), コスト関数:J(w)= (1/2)*∑( y-Φ(w^T*x) )^2）
![twitter_adaline_1-2_170718](https://user-images.githubusercontent.com/25688193/28357349-152a9656-6ca6-11e7-9611-90643928b4a6.png)

</br>

---

<a name="#背景理論"></a>

## 背景理論

<a name="#背景理論１"></a>

## 背景理論１

<a name="#背景理論２"></a>

## 背景理論２
