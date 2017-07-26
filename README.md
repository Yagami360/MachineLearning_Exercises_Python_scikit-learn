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

![perceptron_scikit-learn_2](https://user-images.githubusercontent.com/25688193/28395827-d3c43ef6-6d31-11e7-9421-0fb406a6ec49.png)

データの分割に使用した、交差確認法（クロスバリデーション）について

![twitter_ 5-2_160919](https://user-images.githubusercontent.com/25688193/28366331-2ee5c04a-6cc7-11e7-9085-210c9b0de274.png)


## ./LogisticRegression_scikit-learn
![twitter_ 18-1_161130](https://user-images.githubusercontent.com/25688193/28620065-596c98e6-7246-11e7-86bc-b162dc67923f.png)
![twitter_ 18-2_161130](https://user-images.githubusercontent.com/25688193/28620079-6469d858-7246-11e7-8228-38d902db541f.png)
![twitter_ 18-3_161130](https://user-images.githubusercontent.com/25688193/28620103-7ce82c9a-7246-11e7-80c1-0de312e10d62.png)
![twitter_ 18-4_161130](https://user-images.githubusercontent.com/25688193/28620114-8273782c-7246-11e7-8505-4627605c4290.png)
![twitter_ 18-5_161201](https://user-images.githubusercontent.com/25688193/28620120-884b54a4-7246-11e7-819a-61e546a75fb1.png)
![twitter_ 18-6 _170204](https://user-images.githubusercontent.com/25688193/28620130-93ed169e-7246-11e7-9bea-de868a82455b.png)
![twitter_ 18-7 _170204](https://user-images.githubusercontent.com/25688193/28620136-9846c1f4-7246-11e7-88af-7ac885a097e9.png)

![twitter_ 18-17_170726](https://user-images.githubusercontent.com/25688193/28604784-47ddf5f4-7208-11e7-8136-3ac637f584f2.png)

![twitter_ 18-18_170726](https://user-images.githubusercontent.com/25688193/28615080-dcc565d8-7232-11e7-9e72-d7a9b8166136.png)

<機械学習＆Pythonの練習Memo>

scikit-learn ライブラリの sklearn.linear_model モジュールの LogisticRegression クラスのpredict_proba() 関数を使用して、指定したサンプルのクラスの所属確率を予想

![logisticregression_scikit-learn_4](https://user-images.githubusercontent.com/25688193/28619864-725f0614-7245-11e7-8534-6c162eba8dd3.png)
