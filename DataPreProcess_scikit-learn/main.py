# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

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
    prePro1.print()

    prePro1.meanImputationNaN()
    prePro1.print()

    #--------------------------------------------------
    # Practice 2 : カテゴリデータの処理
    # 名義 [nominal] 特徴量、順序 [ordinal] 特徴量
    #--------------------------------------------------
    prePro2 = DataPreProcess.DataPreProcess()

    prePro2.setDataFrame(
        [
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']
        ]
    )

    #prePro2.setColumns( columns = ['color', 'size', 'price', 'classlabel'] )
    prePro2.print()
    



    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

