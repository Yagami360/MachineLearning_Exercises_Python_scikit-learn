## 主成分分析 [PCA : Principal Component Analysis] による教師なしデータの次元削除、特徴抽出

## コードの実行結果

### Wine データセット (https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)

13×178 次元のデータ（13：特徴量、178：データ数）

||Class label  |Alcohol  |Malic acid   |Ash  |Alcalinity of ash|Magnesium |Total phenols|Flavanoids|Nonflavanoid phenols|Proanthocyanins|Color intensity|Hue|OD280/OD315 of diluted wines|Proline|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0 |1 |14.23 |1.71  |2.43 |15.6 |127 |2.80|3.06|0.28|2.29|5.640000|1.04|3.92|1065|
|1 |1 |13.20 |1.78  |2.14 |11.2 |100 |2.65|2.76|0.26|1.28|4.380000|1.05|3.40|1050|
|2 |1 |13.16 |2.36  |2.67 |18.6 |101 |2.80|3.24|0.30|2.81|5.680000|1.03|3.17|1185|
|3 |1 |14.37 |1.95  |2.50 |16.8 |113 |3.85|3.49|0.24|2.18|7.800000|0.86|3.45|1480|
|4 |1 |13.24 |2.59  |2.87 |21.0 |118 |2.80|2.69|0.39|1.82|4.320000|1.04|2.93|735|  
|5 |1 |14.20 |1.76  |2.45 |15.2 |112 |3.27|3.39|0.34|1.97|6.750000|1.05|2.85|1450|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|170|3|12.20|3.03|2.32|19.0|96 |1.25|0.49|0.40|0.73|5.500000  |0.66|1.83|510|   
|171|3|12.77|2.39|2.28|19.5|86 |1.39|0.51|0.48|0.64|9.899999  |0.57|1.63|470|  
|172|3|14.16|2.51|2.48|20.0|91 |1.68|0.70|0.44|1.24|9.700000  |0.62|1.71|660|  
|173|3|13.71|5.65|2.45|20.5|95 |1.68|0.61|0.52|1.06|7.700000  |0.64|1.74|740|  
|174|3|13.40|3.91|2.48|23.0|102|1.80|0.75|0.43|1.41|7.300000  |0.70|1.56|750|  
|175|3|13.27|4.28|2.26|20.0|120|1.59|0.69|0.43|1.35|10.200000 |0.59|1.56|835|  
|176|3|13.17|2.59|2.37|20.0|120|1.65|0.68|0.53|1.46|9.300000  |0.60|1.62|840| 
|177|3|14.13|4.10|2.74|24.5|96 |2.05|0.76|0.56|1.35|9.200000  |0.61|1.60|560| 


### 固有値 [eigenvalue] 

|λ_1|λ_2|λ_3|λ_4|λ_5|λ_6|λ_7|λ_8|λ_9|λ_10|λ_11|λ_12|λ_13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|4.56|2.65|1.33|1.13|0.80|0.55|0.43|0.25|0.22|0.18|0.16|0.12|0.11|

![pca_scikit-learn_1](https://user-images.githubusercontent.com/25688193/29246419-1b9440ae-8034-11e7-979c-566d42c37b5f.png)

### 寄与率（分散の比）[proportion of the variance] / 累積寄与率 [Cumulative contribution rate]

|principal component|1|2|3|4|5|6|7|8|9|10|11|12|13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|寄与率|0.366|0.213|0.107|0.090|0.064|0.044|0.035|0.020|0.017|0.014|0.0125|0.009|0.0086|
|累積寄与率|0.366|0.578|0.685|0.775|0.839|0.883|0.918|0.938|0.955|0.970|0.982|0.991|1.000|

![pca_scikit-learn_2](https://user-images.githubusercontent.com/25688193/29246420-1ecbdf3e-8034-11e7-9732-1979c1d9c597.png)

### 13×178 次元のワインデータ → 2×124 次元のデータに次元削除（特徴抽出）（※124は分割したトレーニングデータ数）

ワインデータをPCAによる次元削除を行なったデータの散布図。
寄与率と累積寄与率の図より、第１主成分と第２主成分だけで、全体のデータの６０％近くを説明できることから、2×124 次元のデータで散布図を図示。この後、この次元削除したデータでクラス識別用のデータに使用する。

![pca_scikit-learn_3](https://user-images.githubusercontent.com/25688193/29248635-f33244ac-8057-11e7-9de8-89b925f16560.png)

### 次元削除したトレーニングデータをロジスティクス回帰で識別

|classifier 1 : logisitic Regression 1 ( leraning data dimesion by PCA  )|value|
|---|--|
|誤識別数 [Misclassified samples]|4|
|正解率 [Accuracy]|0.97|

![pca_scikit-learn_4](https://user-images.githubusercontent.com/25688193/29248838-06c82450-805d-11e7-8dc8-a3a2db61e9f6.png)

### 次元削除したテストデータをロジスティクス回帰で識別

|classifier 2 : logisitic Regression 2 ( test data dimesion by PCA  )|value|
|---|--|
|誤識別数 [Misclassified samples]|1|
|正解率 [Accuracy]|0.98|


![pca_scikit-learn_5](https://user-images.githubusercontent.com/25688193/29248839-0ba60bd6-805d-11e7-84f4-66ee85f076cb.png)

## Theory
![twitter_ _1-1_170812](https://user-images.githubusercontent.com/25688193/29239290-5fb880fa-7f86-11e7-8ccf-a4d5b7d5cb93.png)

PCAの分散最大化による定式化

![twitter_pca_1-1_170812](https://user-images.githubusercontent.com/25688193/29239293-62991898-7f86-11e7-9f89-eb3b8fcd02a9.png)
![twitter_pca_1-2_170812](https://user-images.githubusercontent.com/25688193/29240813-9eac90e6-7fa7-11e7-9205-836d275f4d64.png)
![twitter_pca_1-3_170813](https://user-images.githubusercontent.com/25688193/29246918-bf1356c2-8041-11e7-9bd8-0c708c562d4c.png)

PCAの誤差最小化による定式化

![twitter_ _ 1-1_161209](https://user-images.githubusercontent.com/25688193/29246920-c49d31d0-8041-11e7-8f4f-10d130c6c370.png)
![twitter_ _ 1-2_161209](https://user-images.githubusercontent.com/25688193/29246921-c49d4440-8041-11e7-950a-b26f7308112f.png)
![twitter_ _ 1-3_161209](https://user-images.githubusercontent.com/25688193/29246922-c49d4da0-8041-11e7-9688-a415b282b863.png)
![twitter_ _ 1-4_161209](https://user-images.githubusercontent.com/25688193/29246924-c49e1258-8041-11e7-84d2-351415a2b231.png)
![twitter_ _ 1-5_161209](https://user-images.githubusercontent.com/25688193/29246923-c49e004c-8041-11e7-9ce9-615cd35a1ed9.png)
![twitter_ _ 1-6_161209](https://user-images.githubusercontent.com/25688193/29246925-c4bf99c8-8041-11e7-9a04-20162af35f68.png)
![twitter_ _ 1-7_161209](https://user-images.githubusercontent.com/25688193/29246927-c4bff49a-8041-11e7-9c40-f054884f774e.png)
![twitter_ _ 1-8_161209](https://user-images.githubusercontent.com/25688193/29246926-c4bfbe26-8041-11e7-8243-bde1a5d99b93.png)
![twitter_ _ 1-9_161209](https://user-images.githubusercontent.com/25688193/29246928-c4c059bc-8041-11e7-843f-177faae564c5.png)
![twitter_ _ 1-10_161209](https://user-images.githubusercontent.com/25688193/29246919-c49cf012-8041-11e7-8102-e04e4d4e1a80.png)
