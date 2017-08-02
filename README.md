# MachineLearning_Samples_Python
機械学習のサンプルコード集（Python使用）。機械学習とPythonのお練習。

README.md：コード実行結果の解説と合わせて、理論の解説も記載してます。

### 目次（フォルダ）

1. [./Perceptron](#Perceptron)
1. [./AdaLineGD](#AdaLineGD) 
1. [./AdaLineSGD](#AdaLineSGD)
1. [./Perceptron_scikit-learn](#Perceptron_scikit-learn)
1. [./LogisticRegression_scikit-learn](#LogisticRegression_scikit-learn)
1. [./SVM_scikit-learn](#SVM_scikit-learn)
1. [./kNN_scikit-learn](#kNN_scikit-learn)
1. [./DecisionTree_scikit-learn](#DecisionTree_scikit-learn)
1. [./RandomForests_scikit-learn](#RandomForests_scikit-learn)

<a name="Perceptron"></a>
## ./Perceptron 

<機械学習＆Pythonの練習Memo>

アヤメデータを単一パーセプトロン＆最急降下法で識別（重みの更新:Δw = η*(y_i-y^_i)）

![twitter_ _1_2_170718](https://user-images.githubusercontent.com/25688193/28357345-0fc51218-6ca6-11e7-859e-5e1d71bca1c2.png)


<a name="AdaLineGD"></a>
## ./AdaLineGD

<機械学習＆Pythonの練習Memo>

アヤメデータをAdaLine＆最急降下法（コスト関数）でのバッチ学習で識別（重みの更新:Δw=η*∑( y-Φ(w^T*x) ) (j=1,2,...,m), コスト関数:J(w)= (1/2)*∑( y-Φ(w^T*x) )^2）

![twitter_adaline_1-2_170718](https://user-images.githubusercontent.com/25688193/28357349-152a9656-6ca6-11e7-9611-90643928b4a6.png)


<a name="AdaLineSGD"></a>
## ./AdaLineSGD

<機械学習＆Pythonの練習Memo> 

アヤメデータをAdaLine＆確率的最急降下法（コスト関数）、及びオンライン学習で識別（重みの更新：Δw=η*( y_i - Φ(w^T*x_i) ), J=(1/2)*( y_i - Φ(w^T*x_i) )^2, i：ランダム）

![twitter_adaline_2-2_170719](https://user-images.githubusercontent.com/25688193/28357356-19940cb8-6ca6-11e7-80ba-50e0c968f6dc.png)


<a name="Perceptron_scikit-learn"></a>
## ./Perceptron_scikit-learn

<機械学習＆Pythonの練習Memo> 

アヤメデータのパーセプトロンによる３クラス（３品種）識別。データの分割はクロス・バディゲーション法。（scikit-learn ライブラリを使用）パーセプトロン如きでは、やはりうまく識別出来ない。

![perceptron_scikit-learn_2](https://user-images.githubusercontent.com/25688193/28395827-d3c43ef6-6d31-11e7-9421-0fb406a6ec49.png)

データの分割に使用した、交差確認法（クロスバリデーション）について

![twitter_ 5-2_160919](https://user-images.githubusercontent.com/25688193/28366331-2ee5c04a-6cc7-11e7-9085-210c9b0de274.png)


<a name="LogisticRegression_scikit-learn"></a>
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

バイアス・分散トレードオフと過学習、及び正則化による過学習への対応

![twitter_ _3-1_160924](https://user-images.githubusercontent.com/25688193/28652178-310a6984-72c1-11e7-994c-d28390cd000b.png)
![twitter_ _3-2_160924](https://user-images.githubusercontent.com/25688193/28652185-381f5356-72c1-11e7-8fae-37e7f77b4e30.png)
![twitter_ _3-3_160924](https://user-images.githubusercontent.com/25688193/28652189-3da74482-72c1-11e7-8050-866a21cba355.png)
![twitter_ _3-4_170727](https://user-images.githubusercontent.com/25688193/28652195-456e0480-72c1-11e7-96c4-c2d9ee8ba273.png)

＜機械学習＆Pythonの練習Memo＞

ロジスティクス回帰の逆正則化パラメータ C の値と正則化の強さの関係（ロジスティクス回帰における、正則化による過学習への対応）正則化の強さを確認するため、重み係数 w と逆正則化パラメータ C の関係を plot
![twitter_ 18-19_170727](https://user-images.githubusercontent.com/25688193/28652198-4b09b560-72c1-11e7-8053-a9e00b280ef8.png)


<a name="SVM_scikit-learn"></a>
## ./SVM_scikit-learn

![twitter_svm_1-1_170211](https://user-images.githubusercontent.com/25688193/28708179-313cdc98-73b6-11e7-985f-8ced8d316ecc.png)
![twitter_svm_1-2_170211](https://user-images.githubusercontent.com/25688193/28708178-313a6daa-73b6-11e7-9817-8621f3cd9985.png)

![twitter_svm_2-1_170212](https://user-images.githubusercontent.com/25688193/28708177-31342c92-73b6-11e7-9b19-0a41a4b7b705.png)
![twitter_svm_2-2_170212](https://user-images.githubusercontent.com/25688193/28708175-312ab5c2-73b6-11e7-8617-37b57c475b35.png)

![twitter_svm_3-1_170214](https://user-images.githubusercontent.com/25688193/28708174-311e33d8-73b6-11e7-82e5-3da320e93b89.png)
![twitter_svm_3-2_170214](https://user-images.githubusercontent.com/25688193/28708173-311dbda4-73b6-11e7-832e-bf7162703056.png)
![twitter_svm_3-3_170214](https://user-images.githubusercontent.com/25688193/28708172-3118eeaa-73b6-11e7-960a-71824390bee5.png)
![twitter_svm_3-4_170214](https://user-images.githubusercontent.com/25688193/28708171-3113dc62-73b6-11e7-9140-f4974f44b495.png)
![twitter_svm_3-5_170216](https://user-images.githubusercontent.com/25688193/28708170-31097290-73b6-11e7-8d0c-8087e1751fb1.png)

![twitter_svm_4-1_170216](https://user-images.githubusercontent.com/25688193/28708169-310200aa-73b6-11e7-8492-41e07ad0a3f9.png)
![twitter_svm_4-2_170217](https://user-images.githubusercontent.com/25688193/28708168-30faf92c-73b6-11e7-987b-996e874fb16f.png)
![twitter_svm_4-3_170217](https://user-images.githubusercontent.com/25688193/28708165-30eb1a5c-73b6-11e7-8530-e19ac4cef9e1.png)
![twitter_svm_4-4_170218](https://user-images.githubusercontent.com/25688193/28708167-30f916ac-73b6-11e7-976d-d4c1e3a52524.png)
![twitter_svm_4-5_170218](https://user-images.githubusercontent.com/25688193/28708166-30f5c588-73b6-11e7-9d9b-54d46b8a69f5.png)

![twitter_svm_5-1_170219](https://user-images.githubusercontent.com/25688193/28708164-30e4d688-73b6-11e7-89a0-d78b5065b467.png)
![twitter_svm_5-2_170220](https://user-images.githubusercontent.com/25688193/28708163-30def074-73b6-11e7-8d17-57fdbf9bab59.png)
![twitter_svm_5-2_170225](https://user-images.githubusercontent.com/25688193/28708162-30c28aba-73b6-11e7-8e63-aa1d77db8c00.png)
![twitter_svm_5-3_170222](https://user-images.githubusercontent.com/25688193/28708159-30bd4c44-73b6-11e7-91bb-c212ab04a7db.png)
![twitter_svm_5-4_170225](https://user-images.githubusercontent.com/25688193/28708161-30c06262-73b6-11e7-88bd-9ea72837d9c8.png)
![twitter_svm_5-5_170303](https://user-images.githubusercontent.com/25688193/28708158-30bc0e1a-73b6-11e7-9fc1-c015e9403def.png)
![twitter_svm_5-6_170303](https://user-images.githubusercontent.com/25688193/28708157-30bbfba0-73b6-11e7-9aba-894974b30167.png)

![twitter_svm_6-1_170728](https://user-images.githubusercontent.com/25688193/28708061-adc48348-73b5-11e7-8cf8-17f3c3a8ba0e.png)
![twitter_svm_6-2 _170728](https://user-images.githubusercontent.com/25688193/28719743-f71ebd8a-73e5-11e7-91cb-476014748aad.png)
![twitter_svm_6-3_170729](https://user-images.githubusercontent.com/25688193/28736123-694aa3d2-7423-11e7-8bba-92fadfdc645c.png)
![twitter_svm_6-4_170729](https://user-images.githubusercontent.com/25688193/28737648-6f478f8c-742a-11e7-9de9-f3f6d619d623.png)

<a name="kNN_scikit-learn"></a>
## ./kNN_scikit-learn

![twitter_ 14-1_161007](https://user-images.githubusercontent.com/25688193/28742174-1d0f13f4-7464-11e7-8cc9-1d669f2c50ca.png)
![twitter_ 14-2_161007](https://user-images.githubusercontent.com/25688193/28742169-1d0c9bce-7464-11e7-97c2-0ec640aa3e15.png)
![twitter_ 14-3_161008](https://user-images.githubusercontent.com/25688193/28742170-1d0d1270-7464-11e7-8cfb-5ec25983427f.png)
![twitter_ 14-4_161009](https://user-images.githubusercontent.com/25688193/28742171-1d0e1530-7464-11e7-8e32-04b007727098.png)
![twitter_ 14-5_161010](https://user-images.githubusercontent.com/25688193/28742173-1d0f097c-7464-11e7-8df7-cd6018620fbf.png)

![twitter_ 16-1_161011](https://user-images.githubusercontent.com/25688193/28742172-1d0edbfa-7464-11e7-8e82-238a91faf92e.png)
![twitter_ 16-2_161012](https://user-images.githubusercontent.com/25688193/28742176-1d2fe52a-7464-11e7-825d-6d49ca8ccfed.png)

![twitter_ 16-5_161112](https://user-images.githubusercontent.com/25688193/28742175-1d2f1b0e-7464-11e7-9b18-3d74ddd6e142.png)
![twitter_ 16-6_161112](https://user-images.githubusercontent.com/25688193/28742177-1d31eb68-7464-11e7-8bd6-a9443593c392.png)

![twitter_ 16-7_170729](https://user-images.githubusercontent.com/25688193/28742632-1482008c-7470-11e7-9590-df87069db4ed.png)


<a name="DecisionTree_scikit-learn"></a>
## ./DecisionTree_scikit-learn

![twitter_ 21-1_170730](https://user-images.githubusercontent.com/25688193/28753082-4ad10f58-7569-11e7-82b3-8adfae7e562c.png)
![twitter_ 21-2_170730](https://user-images.githubusercontent.com/25688193/28753081-4acca2ba-7569-11e7-9a06-33e59c20fb66.png)
![twitter_ 21-3_170731](https://user-images.githubusercontent.com/25688193/28761658-3b50e77e-75eb-11e7-99bf-fbfe15554aa0.png)
![twitter_ 21-5_170731](https://user-images.githubusercontent.com/25688193/28761659-3b797ed2-75eb-11e7-8c22-9509530bc773.png)
![twitter_ 21-6_170731](https://user-images.githubusercontent.com/25688193/28770902-d6b289ae-761b-11e7-9634-58c09410b7ed.png)

![twitter_ 21-7_170731](https://user-images.githubusercontent.com/25688193/28786076-10325ff6-7653-11e7-99a6-f701b6deda43.png)
![twitter_ 21-8_170801](https://user-images.githubusercontent.com/25688193/28808317-e178537e-76b5-11e7-8358-986a433a532a.png)

![twitter_ 21-9_170801](https://user-images.githubusercontent.com/25688193/28813700-649d371e-76d5-11e7-9044-1e841481367e.png)
![twitter_ 21-10_170801](https://user-images.githubusercontent.com/25688193/28819004-cc4a33a0-76e7-11e7-8816-b8f4bd2b31ac.png)
![twitter_ 21-11_170801](https://user-images.githubusercontent.com/25688193/28819013-cf690e94-76e7-11e7-86cd-ef445a17d7ce.png)

＜Note＞ 決定木の結果の解析追記[17/08/02]
  
・識別境界は、特徴軸に対して平行（他の軸に対して垂直）になるように描画されていることが、逆に言えばこれは単に軸に平行な決定境界では分けられないデータは分けられないと見ることもできる。

・また、評価指数に不純度を使っているので、この不純度がどのような分割でも下がらないような複雑なデータではうまく識別できない。

・３番目の添付画像のように、構築した決定木を図示することが出来る。このようにビジュアル的にモデルの中身を見ることが出来るのが決定木の１つの強み

![twitter_ 21-12_170802](https://user-images.githubusercontent.com/25688193/28858808-38dee410-778e-11e7-89a8-2f993e20a3d1.png)
![twitter_ 21-13_170802](https://user-images.githubusercontent.com/25688193/28861308-39ed005a-779b-11e7-96e1-2e80becc6e82.png)
![twitter_ 21-14_170802](https://user-images.githubusercontent.com/25688193/28867517-5710c002-77b1-11e7-8d12-941e6272d5b4.png)
![twitter_ 21-15_170802](https://user-images.githubusercontent.com/25688193/28867521-5afb3328-77b1-11e7-8b57-ec9e9dc9f255.png)


<a name="RandomForests_scikit-learn"></a>
## ./RandomForests_scikit-learn

![twitter_ 22-1_170802](https://user-images.githubusercontent.com/25688193/28871164-5482daca-77be-11e7-8732-67253307f2c3.png)
![twitter_ 22-2_170802](https://user-images.githubusercontent.com/25688193/28872136-ca4c3a7c-77c2-11e7-972f-4c93662f3ee7.png)
