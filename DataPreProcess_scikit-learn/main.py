# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt
import pandas

# scikit-learn ライブラリ関連
from sklearn import datasets                            # 
#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.metrics import accuracy_score              # 

# 自作クラス
import DataPreProcess


def main():
    #=========================================
    # 機械学習における前処理の練習プログラム
    #=========================================
    print("Enter main()")

    #-----------------------------------------
    # Practice 1 : 欠損値 NaN の補完
    #-----------------------------------------
    prePro1 = DataPreProcess.DataPreProcess()

    csv_data = '''
                  A,B,C,D
                  1.0,2.0,3.0,4.0
                  5.0,6.0,,8.0
                  10.0,11.0,12.0,
               '''
    
    prePro1.setDataFrameFromCsvData( csv_data )
    prePro1.print( "csv data" )

    prePro1.meanImputationNaN()
    prePro1.print( "欠損値 NaN の平均値補完" )

    #--------------------------------------------------
    # Practice 2 : カテゴリデータの処理
    # 名義 [nominal] 特徴量、順序 [ordinal] 特徴量
    #--------------------------------------------------
    prePro2 = DataPreProcess.DataPreProcess()

    # list から pandas データフレームを作成
    prePro2.setDataFrame(
        dataFrame= [ 
                ['green', 'M', 10.1, 'class1'], 
                ['red', 'L', 13.5, 'class2'], 
                ['blue', 'XL', 15.3, 'class1'] 
            ]
    )

    prePro2.print( "list から pandas データフレームを作成" )

    # pandas データフレームにコラム（列）を追加
    prePro2.setColumns( ['color', 'size', 'price', 'classlabel'] )
    prePro2.print( "pandas データフレームにコラム（列）を追加" )
    
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
    prePro2.print( "クラスラベルのエンコーディング（ディクショナリマッピング方式" )

    # カテゴリデータのone-hot encoding
    prePro2.OneHotEncode( categories = ['color', 'size', 'price'], col = 0 )
    prePro2.print( "カテゴリデータのone-hot encoding" )


    #--------------------------------------------------
    # Practice 3 : データセットの分割
    # トレーニングデータとテストデータへの分割
    #--------------------------------------------------
    prePro3 = DataPreProcess.DataPreProcess()

    # Wine データセットの読み込み
    prePro3.setDataFrameFromCsvFile( "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" )
    
    # 列名をセット
    prePro3.setColumns( 
        [
            'Class label', 'Alcohol', 'Malic acid', 'Ash',
            'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
            'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
            'Proline'
        ] 
    )

    prePro3.print("Wine データセット")


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

