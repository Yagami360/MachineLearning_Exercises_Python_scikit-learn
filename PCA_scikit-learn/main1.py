# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets                            # 

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    print("Enter main()")
    #=============================================================================================
    # 主成分分析 [PCA : Principal Component Analysis] による教師なしデータの次元削除、特徴抽出
    # scikit-learn ライブラリでの主成分分析不使用
    #=============================================================================================

    #====================================================
    #   Data Preprocessing（前処理）
    #====================================================
    #----------------------------------------------------
    #   read & set  data
    #----------------------------------------------------
    prePro = DataPreProcess.DataPreProcess()

    # Wine データセットの読み込み
    prePro.setDataFrameFromCsvFile( "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" )
    
    # 上記URLのWine データセットにはラベルがついてないので, 列名をセット
    prePro.setColumns( 
        [
            'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
            'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
            'Proline'
        ] 
    )

    prePro.print("Wine データセット")

    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( 
        X_input = prePro.df_.iloc[:, 1:].values,   # iloc : 行、列を番号で指定（先頭が 0）。df_.iloc[:, 1:] = 全行、1~の全列
        y_input = prePro.df_.iloc[:, 0].values,    #
        ratio_test = 0.3
    )

    # 分割データ（トレーニングデータ、テストデータ）を出力
    print( "トレーニングデータ : \n", X_train )
    print("テストデータ : \n", X_test )
    print("トレーニング用教師データ : \n", y_train )
    print("テスト用教師データ : \n", y_test )

    # 特徴量のスケーリング（標準化 [standardization]）
    X_train_std, X_test_std \
    = DataPreProcess.DataPreProcess.standardizeTrainTest( X_train, X_test )

    # 標準化後のデータを出力
    print( "トレーニングデータ [standardized] :\n", X_train_std )
    print("テストデータ [standardized] : \n", X_test_std )
    
    #====================================================
    #   Learning Process
    #====================================================
    

    #====================================================
    #   汎化性能の評価
    #====================================================

    

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
