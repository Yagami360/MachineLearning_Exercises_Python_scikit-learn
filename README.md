# MachineLearning_Samples_Python
機械学習のサンプルコード集（Python使用）。機械学習とPythonのお練習（初歩）

## ./Perceptron

<機械学習＆Pythonの練習Memo>

アヤメデータを単一パーセプトロン＆最急降下法で識別（重みの更新:Δw = η*(y_i-y^_i)）

![twitter_ _1_2_170718](https://user-images.githubusercontent.com/25688193/28357345-0fc51218-6ca6-11e7-859e-5e1d71bca1c2.png)


## ./AdaLineGD

<機械学習＆Pythonの練習Memo>

アヤメデータをAdaLine＆最急降下法（コスト関数）でのバッチ学習で識別（重みの更新:Δw=η*∑( y-Φ(w^T*x) ) (j=1,2,...,m), コスト関数:J(w)= (1/2)*∑( y-Φ(w^T*x) )^2）

![twitter_adaline_1-2_170718](https://user-images.githubusercontent.com/25688193/28357349-152a9656-6ca6-11e7-9611-90643928b4a6.png)


## ./AdaLineSGD

<機械学習＆Pythonの練習Memo> 

アヤメデータをAdaLine＆確率的最急降下法（コスト関数）、及びオンライン学習で識別（重みの更新：Δw=η*( y_i - Φ(w^T*x_i) ), J=(1/2)*( y_i - Φ(w^T*x_i) )^2, i：ランダム）

![twitter_adaline_2-2_170719](https://user-images.githubusercontent.com/25688193/28357356-19940cb8-6ca6-11e7-80ba-50e0c968f6dc.png)


## ./Perceptron_scikit-learn

<機械学習＆Pythonの練習Memo> 

アヤメデータのパーセプトロンによる３クラス（３品種）識別。データの分割はクロス・バディゲーション法。（scikit-learn ライブラリを使用）パーセプトロン如きでは、やはりうまく識別出来ない。

![twitter_python_scikit-learn_1_1_170719](https://user-images.githubusercontent.com/25688193/28357359-1cca73fe-6ca6-11e7-97e3-f3d971c96501.png)

データの分割に使用した、交差確認法（クロスバリデーション）について

![twitter_ 5-3_160922](https://user-images.githubusercontent.com/25688193/28366250-daca3504-6cc6-11e7-83eb-64bb9e5de270.png)

