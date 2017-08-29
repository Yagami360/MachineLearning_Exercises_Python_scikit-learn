## ランダムフォレスト [RandomForest]

### 項目 [Contents]

1. [使用するライブラリ](#使用するライブラリ)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](#背景理論)

</br>

<a name="#使用するライブラリ"></a>

### 使用するライブラリ：

>ランダムフォレスト : `sklearn.ensemble.RandomForestClassifier`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html</br>

<a name="#コードの実行結果"></a>

## コードの実行結果

ランダムフォレストを用いた識別問題における、森のサイズと特徴の誤り率＆OOB誤り率の変化の関係図。

データはアヤメデータの３品種、合計150個を、それぞれ指定された森のサイズ（弱識別器の数）を持つランダムフォレスト全てに適用している。
決定木構築のための不純度としてジニ係数を使用。

森のサイズ（弱識別器数：決定木数）が大きくなるほど誤り率は低下しており、およそ２５個程度で値が収束（OOB error rate : 0.057, iris error rate : 0.002）していることが分かる。
![ramdomforest_scikit-learn_3](https://user-images.githubusercontent.com/25688193/28994775-21988056-7a11-11e7-9e6c-780369f19ebf.png)

---

ランダムフォレストを用いたアヤメデータの識別問題における、各特徴（がく片 [sepal] の長さ、幅、花弁 [petal] の長さ、幅）の関係図。

データはアヤメデータの３品種合計150個を、50個の森を持つランダムフォレストに適用。

データは、トレーニング用：１０５個、テスト用：４５個、に分割する。データの分割は、トレーニングデータ：70%, テストデータ :30% のランダムサンプリング。決定木構築のための不純度としてジニ係数を使用。

＜各特徴の重要度の値＞

1 : sepal length (cm)  = 0.151

2 : sepal width (cm)  = 0.025

3 : petal length (cm) = 0.429010

4 : petal width (cm) =0.395

![ramdomforest_scikit-learn_4](https://user-images.githubusercontent.com/25688193/29001389-54ede3c0-7ac4-11e7-9cac-1599d07bf28e.png)


<a name="#背景理論"></a>

## 背景理論

![twitter_ 22-1_170802](https://user-images.githubusercontent.com/25688193/28871164-5482daca-77be-11e7-8732-67253307f2c3.png)
![twitter_ 22-2_170802](https://user-images.githubusercontent.com/25688193/28893585-f645aae2-780c-11e7-9c25-47b9b92e5017.png)
![twitter_ 22-3_170802](https://user-images.githubusercontent.com/25688193/28875951-68509326-77d2-11e7-89d8-dbc5388193f8.png)

![twitter_ 22-4_170803](https://user-images.githubusercontent.com/25688193/28907039-f5c7e96e-7856-11e7-902d-d1aaaba7954c.png)
![twitter_ 22-5_170803](https://user-images.githubusercontent.com/25688193/28907040-f78f779e-7856-11e7-839d-9845e29dce24.png)

![twitter_ 22-6_170804](https://user-images.githubusercontent.com/25688193/28962339-5fc9099e-7940-11e7-8693-9e7d5019acb7.png)
![twitter_ 22-7_170804](https://user-images.githubusercontent.com/25688193/28963702-0961b768-7945-11e7-8043-8c2d24884d44.png)
![twitter_ 22-8_170804](https://user-images.githubusercontent.com/25688193/28965432-fc909656-794b-11e7-9c16-3c34c381c8d9.png)


