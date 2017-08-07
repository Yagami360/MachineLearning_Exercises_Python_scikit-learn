<a name="DataPreProcess_scikit-learn"></a>
# DataPreProcess_scikit-learn

機械学習における、データの前処理のサンプルコード集。（練習プログラム）

pandas ライブラリ、scikit-learn ライブラリを使用。

## Practice 1 : 欠損値 NaN への対応

#### ・欠損値 NaNの補完（平均値）

csv data (欠損値 NaN を含むデータ)

||A|B|C|D|
|:--:|:--:|:--:|:--:|:--:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|NaN|8.0|
|2|10.0|11.0|12.0|NaN|

欠損値 NaN の平均値補完

```
import numpy
import pandas
from sklearn.preprocessing import Imputer

class DataPreProcess( object ):
    ....
    
    def meanImputationNaN( self, axis = 0 ):
        """
        欠損値 [NaN] を平均値で補完する
        [Input]
            axis : int
                0 : NaN を列の平均値で補完
                1 : NaN を行の平均値で補完
        """
        imputer = Imputer( 
                      missing_values = 'NaN', 
                      strategy = 'mean', 
                      axis = axis       # 0 : 列の平均値, 1 : 行の平均値
                  )
        
    imputer.fit( self.df_ )         # self.df_ は１次配列に変換されることに注意
    self.df_ = imputer.transform( self.df_ )
    return self
```

||A|B|C|D|
|:--:|:--:|:--:|:--:|:--:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|7.5|8.0|
|2|10.0|11.0|12.0|6.0|

## Practice 2 : カテゴリデータ（名義 [nominal] 特徴量、順序 [ordinal] 特徴量）の処理

#### ・名義 [nominal] 特徴量の map(directionary) での整数化
#### ・クラスラベルのエンコーディング（ディクショナリマッピング方式）

list から pandas データフレームを作成

||0|1|2|3|
|---|---|---|---|---|
|0 |green |M   |10.1  |class1|
|1 |red   |L   |13.5  |class2|
|2 |blue  |XL  |15.3  |class1|

pandas データフレームにコラム（列）を追加

||color |size  |price |classlabel|
|---|---|---|---|---|
|0  |green |M   |10.1 |class1|
|1  |red   |L   |13.5 |class2|
|2  |blue  |XL  |15.3 |class1|

順序特徴量 size の map(directionary) を作成し、作成した map で順序特徴量を整数化

```
import numpy
import pandas
...

class DataPreProcess( object ):
...

    def EncodeClassLabel( self, key ):
        """
        クラスラベルを表す文字列を 0,1,2,.. の順に整数化する.（ディクショナリマッピング方式）
        [Input]
            key : string
                整数化したいクラスラベルの文字列
        """
        mapping = { label: idx for idx, label in enumerate( numpy.unique( self.df_[key]) ) }
        self.df_[key] = self.df_[key].map( mapping )

        return self

def main():
    ....
    # 順序特徴量 size の map(directionary) を作成
    dict_size = {
        'XL': 3,
        'L': 2,
        'M': 1
    }
    # 作成した map で順序特徴量を整数化
    prePro2.MappingOrdinalFeatures( key = 'size', input_dict = dict_size )
    prePro2.print( "順序特徴量 size の map(directionary) を作成し、作成した map で順序特徴量を整数化" )
    
    # クラスラベルのエンコーディング（ディクショナリマッピング方式）
    prePro2.EncodeClassLabel("classlabel")
```

||color  |size  |price |classlabel|
|---|---|---|---|---|
|0  |green  |1 |10.1  |class1 |
|1  |red    |2 |13.5  |class2 |
|2  |blue   |3 |15.3  |class1 |

#### ・カテゴリデータの one-hot encoding

|size|price|color_blue|color_green|color_red|
|---|---|---|---|---|
|0|1|10.1|0|1|0|
|1|2|13.5|0|0|1|
|2|3|15.3|1|0|0|


## Practice 3 : データセットの分割

#### ・トレーニングデータとテストデータへの分割とその割合

## Practice 4 : 特徴量のスケーリング

#### ・正規化 [nomalization]（min-max スケーリング <0~1>）

#### ・標準化 [standardization]（平均値：０、分散値：１）

## Practice 4 : 有益な特徴量の選択

#### ・L1正則化による疎な解
