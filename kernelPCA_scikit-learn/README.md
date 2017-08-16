## カーネル主成分分析 [kernelPCA : kernel Principal Component Analysis] による非線形写像と教師なしデータの次元削除、特徴抽出



## コードの実行結果

### `main1.py`

`sklearn.datasets.make_moons( n_samples = 100 )` で生成した半月状のデータ（サンプル数：１００個）に対し、通常の PCA を適用した結果。
上段の図より、通常の PCA では、うまく線形分離可能なテータに変換できていないことが分かる。
（※下段の図は、各主成分に対する固有値と寄与率、累積率寄与率の図）

![kernelpca_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29361298-eef3455a-82c1-11e7-8329-04792751bf7e.png)

`sklearn.datasets.make_moons( n_samples = 100 )` で生成した半月状のデータ（サンプル数：１００個）に対し、RBF カーネルをカーネル関数とする、カーネル PCA を適用した結果。
RBF-kernel PCA では、うまく線形分離可能なテータに変換できていることが分かる。

![kernelpca_scikit-learn_2](https://user-images.githubusercontent.com/25688193/29361299-f106740c-82c1-11e7-9596-74a0c7da535a.png)


## Theory

![twitter_pca_2-1_170815](https://user-images.githubusercontent.com/25688193/29283593-621e79b6-8162-11e7-8624-e5c914da21f6.png)
![twitter_pca_2-2_170815](https://user-images.githubusercontent.com/25688193/29303785-3cead10e-81ca-11e7-9ffd-46aa36d8869e.png)
![twitter_pca_2-3_170815](https://user-images.githubusercontent.com/25688193/29308244-00c0b96c-81e0-11e7-913c-f8c2ec4f80ed.png)
![twitter_pca_2-4_170815](https://user-images.githubusercontent.com/25688193/29308248-052d58a2-81e0-11e7-94cf-57018daecce2.png)
