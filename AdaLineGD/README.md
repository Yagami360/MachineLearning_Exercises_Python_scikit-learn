## AdaLineGD（Adaptive Liner ニューロン勾配降下法） [Adaptive Liner Gradient descent]

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの実行結果](#ID_3)
    1. [AdaLine によるアヤメデータの識別 : `main1.py`](#ID_3-1)
1. [背景理論](#ID_4)
    1. [ニューラルネットワークの概要](#ID_4-1)
    1. [活性化関数](#ID_4-2)
    1. [単純パーセプトロン](#ID_4-3)
    1. [パーセプトロンによる論理演算](#ID_4-4)
    1. [最急降下法による学習](#ID_4-5)


<a id="ID_1"></a>

### 使用するライブラリ


<a id="ID_2"></a>

### 使用するデータセット

> Iris データセット : csv フォーマット </br>
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data


<a id="ID_3"></a>

## コードの実行結果

<a id="ID_3-1"></a>

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
- 自作クラス `AdaLineGD` を用いて、学習データでの fitting 処理（重みベクトルの更新処理）を行う。</br> `AdaLineGD.fit( X_features, y_labels )`

</br>

> アヤメデータをAdaLine＆最急降下法（コスト関数）でのバッチ学習で識別結果。</br>（重みの更新:Δw=η*∑( y-Φ(w^T*x) ) (j=1,2,...,m), コスト関数:J(w)= (1/2)*∑( y-Φ(w^T*x) )^2）
![twitter_adaline_1-2_170718](https://user-images.githubusercontent.com/25688193/28357349-152a9656-6ca6-11e7-9611-90643928b4a6.png)

</br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## ニューラルネットワークの概要
![twitter_nn1_1_160825](https://user-images.githubusercontent.com/25688193/29994077-594d50c4-9002-11e7-829d-5a695503b486.png)
![twitter_nn1_2_160825](https://user-images.githubusercontent.com/25688193/29994078-594deebc-9002-11e7-801f-d0d6617cbde6.png)
![twitter_nn3 -1_160827](https://user-images.githubusercontent.com/25688193/29994081-5976f6cc-9002-11e7-9587-dc3cb098b325.png)

<a id="ID_4-2"></a>

## 活性化関数
![twitter_nn2-1_160826](https://user-images.githubusercontent.com/25688193/29994079-59705a74-9002-11e7-88ba-214af1ceec62.png)
![twitter_nn2-2_160826](https://user-images.githubusercontent.com/25688193/29994080-5970ebe2-9002-11e7-86fb-769349356224.png)

<a id="ID_4-3"></a>

# 単純パーセプトロン
![twitter_nn4 -1_160829](https://user-images.githubusercontent.com/25688193/29994084-598c65c0-9002-11e7-9f9b-a529d44f1f8a.png)

<a id="ID_4-4"></a>

# パーセプトロンによる論理演算
![twitter_nn6-1_160829](https://user-images.githubusercontent.com/25688193/29994082-597791ea-9002-11e7-9bb5-2ae6bc436f56.png)
![twitter_nn6-2_160829](https://user-images.githubusercontent.com/25688193/29994083-598aa280-9002-11e7-9ec0-16316a04686a.png)

<a id="ID_4-5"></a>

# 最急降下法による学習
![image](https://user-images.githubusercontent.com/25688193/30624595-3a3797da-9df9-11e7-95eb-5edb913e080f.png)

<a id="ID_4-5-1"></a>

##### 最急降下法の単層パーセプトロンでの適用
![image](https://user-images.githubusercontent.com/25688193/30637199-9c51d226-9e32-11e7-9301-e9a66ca6e34c.png)
![image](https://user-images.githubusercontent.com/25688193/30637749-38a66b18-9e34-11e7-827a-c282cb8986c2.png)

<a id="ID_4-5-2"></a>

##### 最急降下法の多層パーセプトロンでの適用
![image](https://user-images.githubusercontent.com/25688193/30634693-e32e0104-9e2a-11e7-87d9-8b570b8af3b0.png)
![image](https://user-images.githubusercontent.com/25688193/30634719-f9f57a84-9e2a-11e7-9de0-3d8da1268785.png)
![image](https://user-images.githubusercontent.com/25688193/30636431-65a9b5ec-9e30-11e7-9b83-d3a87daa7513.png)
