## 決定木 [Decision Tree]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [決定木の不純度を表す関数の作図](#決定木の不純度を表す関数の作図)
    1. [決定木（CRAT）による３クラス（アヤメデータ）の識別](#決定木（CRAT）による３クラス（アヤメデータ）の識別)
    1. [決定木の樹形図の表示](#決定木の表示)
1. [背景理論](#背景理論)
    1. [決定木](#決定木)
        1. [決定木に関する諸定義](#決定木に関する諸定義)
    1. [決定木での特徴空間のクラス分け（識別クラス）と確率分布](#決定木での特徴空間のクラス分け（識別クラス）と確率分布)
    1. [ノードの分割規則と不純度](#ノードの分割規則と不純度)

</br>

<a name="#使用するライブラリ"></a>

### 使用するライブラリ

> scikit-learn ライブラリ </br>
>> データセット Dataset loading utilities : `sklearn.datasets`</br>
>> http://scikit-learn.org/stable/datasets/index.html </br>

>> モデル選択 : `sklearn.model_selection` </br>
>>> データの分割 : `sklearn.model_selection.train_test_split()`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html</br>

>> 前処理 : `sklearn.preprocessing` </br>
>>> 正規化処理 :  `sklearn.preprocessing.StandardScaler`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html </br>

>> 正解率の算出 : `sklearn.metrics.accuracy_score` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html </br>

>> 決定木 : `sklearn.tree.DecisionTreeClassifier` </br>
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html</br>
>>> 決定木の樹形図作図 : `sklearn.tree.export_graphviz`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html


</br>

<a name="#使用するデータセット"></a>

### 使用するデータセット

> Iris データセット : </br>
> https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#決定木の不純度を表す関数の作図"></a>

## 決定木の不純度を表す関数の作図 : `main1.py`
- 不純度を表す関数として、</br>ノードの誤り率 [eror rate], 交差エントロピー関数 [cross-entropy], ジニ係数 [Gini index] があるが、</br>これらの 識別クラスが C1 のみのケースでの関数の作図を行う。
- ノードの誤り率の計算式 : </br>
`value = 1 - numpy.max( [p, 1-p] )  #　p : 確率値（0.0 ~ 1.0）`
- 交差エントロピー関数の計算式（２クラスのバイナリーなので log の底が 2）: </br>
`value = - p * numpy.log2(p) - (1 - p) * numpy.log2( (1 - p) )`
- ジニ係数の計算式 : </br>
`value = p * (1 - p) + (1 - p) * ( 1 - (1 - p) )`

![twitter_ 21-8_170801](https://user-images.githubusercontent.com/25688193/28808317-e178537e-76b5-11e7-8358-986a433a532a.png)

<a name="#決定木（CRAT）による３クラス（アヤメデータ）の識別"></a>

## 決定木（CRAT）による３クラス（アヤメデータ）の識別 : `main2.py`
- Iris データセットを使用 : `datasets.load_iris()`
- 特徴行列 `X_features` は、特徴数 2 個 × サンプル数 150 個 :</br> `iris.data[ :, [2,3] ]`
- サンプル数 150 個の内、品種 "setosa" が 50 個、"virginica" が 50 個、"versicolor" が 50 個。
- 教師データ `y_labels` は、サンプル数 150 個 : `y_labels = iris.target`
    - ここで、`iris.target` で取得したデータは、カテゴリーデータを 0, 1, 2 に変換したデータとなっている。
- トレーニングデータ 70% 、テストデータ 30%の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = 0.3, random_state = 0 )`
- 正規化処理を実施 : </br> 
`sklearn.preprocessing.StandardScaler` クラスを使用 
- モデルとして、それぞれハイパーパラメータの異なる３つの 決定木 モデルを使用する。</br>
（グリッドサーチによる最適なハイパーパラメータの検討は行わない）</br>
    - 不純度としてジニ係数を使用 : </br>
`tree1 = DecisionTreeClassifier( criterion = 'gini', max_depth = 2, random_state = 0 )`</br>
`tree2 = DecisionTreeClassifier( criterion = 'gini', max_depth = 3, random_state = 0 )`</br>
`tree3 = DecisionTreeClassifier( criterion = 'gini', max_depth = 5, random_state = 0 )`</br>
- それらの k-NN モデルの fitting 処理でモデルを学習させる :</br>
`tree1.fit( X_train_std, y_train )`</br>
`tree2.fit( X_train_std, y_train )`</br>
`tree3.fit( X_train_std, y_train )`</br>
- predict した結果 `y_predict = tree.predict( X_test_std )` を元に、`accuracy_score()` 関数で、正解率、誤分類のサンプル数を算出。</br>
正解率 : `accuracy_score( y_test, y_predict )`</br>
誤分類数 : `( y_test != y_predict ).sum()`

識別境界は、特徴軸に対して平行（他の軸に対して垂直）になるように描画されていることが、逆に言えばこれは単に軸に平行な決定境界では分けられないデータは分けられないと見ることもできる。</br>
また、評価指数に不純度を使っているので、この不純度がどのような分割でも下がらないような複雑なデータではうまく識別できない。</br>
３番目の添付画像のように、構築した決定木を図示することが出来る。このようにビジュアル的にモデルの中身を見ることが出来るのが決定木の１つの強み</br>
（※尚、以下の挿入図のデータの分割方法の記載に、クロス・バリデーションの記載があるが、実際にはクロス・バリデーションによる各スコアの評価は行なっていない。）
![twitter_ 21-9_170801](https://user-images.githubusercontent.com/25688193/28813700-649d371e-76d5-11e7-9044-1e841481367e.png)
- `predict_proba()` 関数を使用して、指定したサンプルのクラスの所属関係を予想 : </br>
例 : `tree1.predict_proba( X_test_std[0, :].reshape(1, -1) )`
![twitter_ 21-10_170801](https://user-images.githubusercontent.com/25688193/28819004-cc4a33a0-76e7-11e7-8816-b8f4bd2b31ac.png)

<a name="#決定木の樹形図の表示"></a>

## 決定木の樹形図の表示 : `main3.py`
- まず、決定木の樹形図のグラフの為の dot ファイルを出力する : </br>
```
    export_graphviz( 
        tree1, 
        out_file = "DecisionTree1.dot", 
        feature_names = ['petal length', 'petal width'] 
    )
```
- 出力した dot ファイルは、GraphViz ツール内でプロンプトでのコマンド</br>
dot -Tpng DecisionTree.dot -o DecisionTreeGraph.png</br>
で png ファイル化することで決定木の樹形図を可視化できる.

上記の GraphViz ツールを用いて可視化（pngファイル）した、それぞれハイパーパラメータの異なる３つの 決定木 モデルの樹形図。赤枠で囲んだ先頭の petal width <= -0.6129 がノードの分割のための条件となっており、最初の条件は、３つの決定木モデルで全て共通であることが分かる。</br>
最終的なクラスの識別結果は、赤枠で囲んだ value = [x,x,x] で示されている。
![twitter_ 21-11_170801](https://user-images.githubusercontent.com/25688193/28819013-cf690e94-76e7-11e7-86cd-ef445a17d7ce.png)


---

<a name="#背景理論"></a>

## 背景理論

<a name="#決定木"></a>

### 決定木

![twitter_ 21-1_170730](https://user-images.githubusercontent.com/25688193/28753082-4ad10f58-7569-11e7-82b3-8adfae7e562c.png)
![twitter_ 21-2_170730](https://user-images.githubusercontent.com/25688193/28753081-4acca2ba-7569-11e7-9a06-33e59c20fb66.png)

<a name="#決定木に関する諸定義"></a>

### 決定木に関する諸定義
![twitter_ 21-3_170731](https://user-images.githubusercontent.com/25688193/28761658-3b50e77e-75eb-11e7-99bf-fbfe15554aa0.png)

<a name="#決定木での特徴空間のクラス分け（識別クラス）と確率分布"></a>

### 決定木での特徴空間のクラス分け（識別クラス）と確率分布
![twitter_ 21-5_170731](https://user-images.githubusercontent.com/25688193/28761659-3b797ed2-75eb-11e7-8c22-9509530bc773.png)
![twitter_ 21-6_170731](https://user-images.githubusercontent.com/25688193/28770902-d6b289ae-761b-11e7-9634-58c09410b7ed.png)

<a name="#ノードの分割規則と不純度"></a>

### ノードの分割規則と不純度
![twitter_ 21-7_170731](https://user-images.githubusercontent.com/25688193/28786076-10325ff6-7653-11e7-99a6-f701b6deda43.png)
