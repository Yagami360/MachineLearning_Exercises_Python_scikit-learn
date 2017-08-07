<a name="DataPreProcess_scikit-learn"></a>
# DataPreProcess_scikit-learn

機械学習における、データの前処理のサンプルコード集。（練習プログラム）

pandas ライブラリ、scikit-learn ライブラリを使用。

## Practice 1 : 欠損値 NaN への対応

・欠損値 NaNの補完（平均値）
`
<DataPreProcess.DataPreProcess object at 0x0000000002A80470>
csv data
                     A     B     C    D
0                  1.0   2.0   3.0  4.0
1                  5.0   6.0   NaN  8.0
2                 10.0  11.0  12.0  NaN
<DataPreProcess.DataPreProcess object at 0x0000000002A80470>
欠損値 NaN の平均値補完
[[  1.    2.    3.    4. ]
 [  5.    6.    7.5   8. ]
 [ 10.   11.   12.    6. ]]
 `
 
## Practice 2 : カテゴリデータの処理

・名義 [nominal] 特徴量、順序 [ordinal] 特徴量

・名義 [nominal] 特徴量の map(directionary) での整数化

・クラスラベルのエンコーディング（ディクショナリマッピング方式）

・カテゴリデータの one-hot encoding

## Practice 3 : データセットの分割

・トレーニングデータとテストデータへの分割とその割合

## Practice 4 : 特徴量のスケーリング

・正規化 [nomalization]（min-max スケーリング <0~1>）

・標準化 [standardization]（平均値：０、分散値：１）

## Practice 4 : 有益な特徴量の選択

・L1正則化による疎な解
