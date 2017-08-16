## カーネル主成分分析 [kernelPCA : kernel Principal Component Analysis] による非線形写像と教師なしデータの次元削除、特徴抽出

使用する scikit-learn ライブラリ：

`sklearn.decomposition.KernelPCA` : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

`sklearn.decomposition.PCA` : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

## コードの実行結果

### `main1.py`

`sklearn.datasets.make_moons( n_samples = 100 )` で生成した半月状のデータ（サンプル数：１００個）に対し、通常の PCA を適用した結果。

上段の図より、通常の PCA では、うまく線形分離可能なテータに変換できていないことが分かる。（※下段の図は、各主成分に対する固有値と寄与率、累積率寄与率の図）尚、第１主成分 PC1 のみに次元削除（特徴抽出）した図は、各クラス（0 or 1）の識別を見やすくするため、上下に少し移動させている。

![kernelpca_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29363412-e2cbf200-82ca-11e7-97b9-7ac3edce6383.png)

`sklearn.datasets.make_moons( n_samples = 100 )` で生成した半月状のデータ（サンプル数：１００個）に対し、RBF カーネルをカーネル関数とする、カーネル PCA を適用した結果。
上段の図より、RBFカーネルをカーネルとするkernelPCA では、この半月状のデータをうまく線形分離可能な特徴空間に写像出来ていることが分かる。（尚、第１主成分 PC1 のみに次元削除（特徴抽出）した図は、各クラス（0 or 1）の識別を見やすくするため、上下に少し移動させている。）

下段の図は、RBFカーネル関数のカーネル行列（グラム行列）の固有値を、大きい順に 40 個表示した図。カーネル行列の固有値は固有値分解を近似的な数値解析的手法で解いており、0 に近い値の固有値がこれに続いている。

![kernelpca_scikit-learn_2](https://user-images.githubusercontent.com/25688193/29363414-e47126c0-82ca-11e7-8931-10472ac76627.png)


`sklearn.datasets.make_circles( n_samples = 1000 )` で生成した同心円状のデータ（サンプル数：１０００個）に対し、通常の PCA を適用した結果。

上段の図より、通常の PCA では、うまく線形分離可能なテータに変換できていないことが分かる。（※下段の図は、各主成分に対する固有値と寄与率、累積率寄与率の図）尚、第１主成分 PC1 のみに次元削除（特徴抽出）した図は、各クラス（0 or 1）の識別を見やすくするため、上下に少し移動させている。

![kernelpca_scikit-learn_4](https://user-images.githubusercontent.com/25688193/29364831-bc25d8dc-82cf-11e7-84b2-9842d5e96a1c.png)


`sklearn.datasets.make_circles( n_samples = 1000 )` で生成した同心円状のデータ（サンプル数：１０００個）に対し、RBF カーネルをカーネル関数とする、カーネル PCA を適用した結果。
上段の図より、RBFカーネルをカーネルとするkernelPCA では、この半月状のデータをうまく線形分離可能な特徴空間に写像出来ていることが分かる。（尚、第１主成分 PC1 のみに次元削除（特徴抽出）した図は、各クラス（0 or 1）の識別を見やすくするため、上下に少し移動させている。）

![kernelpca_scikit-learn_5](https://user-images.githubusercontent.com/25688193/29364832-bc48a862-82cf-11e7-9e59-25991406e03c.png)


## Theory

![twitter_pca_2-1_170815](https://user-images.githubusercontent.com/25688193/29283593-621e79b6-8162-11e7-8624-e5c914da21f6.png)
![twitter_pca_2-2_170815](https://user-images.githubusercontent.com/25688193/29303785-3cead10e-81ca-11e7-9ffd-46aa36d8869e.png)
![twitter_pca_2-3_170815](https://user-images.githubusercontent.com/25688193/29308244-00c0b96c-81e0-11e7-913c-f8c2ec4f80ed.png)
![twitter_pca_2-4_170815](https://user-images.githubusercontent.com/25688193/29308248-052d58a2-81e0-11e7-94cf-57018daecce2.png)
