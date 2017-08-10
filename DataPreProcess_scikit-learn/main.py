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

# ロジスティクス回帰
from sklearn.linear_model import LogisticRegression

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
    prePro2.setDataFrameFromList(
        list = [ 
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
    prePro2.mappingOrdinalFeatures( key = 'size', input_dict = dict_size )
    prePro2.print( "順序特徴量 size の map(directionary) を作成し、作成した map で順序特徴量を整数化" )
    
    # クラスラベルのエンコーディング（ディクショナリマッピング方式）
    prePro2.encodeClassLabel("classlabel")
    prePro2.print( "クラスラベルのエンコーディング（ディクショナリマッピング方式" )

    # カテゴリデータのone-hot encoding
    prePro2.oneHotEncode( categories = ['color', 'size', 'price'], col = 0 )
    prePro2.print( "カテゴリデータのone-hot encoding" )


    #--------------------------------------------------
    # Practice 3 : データセットの分割
    # トレーニングデータとテストデータへの分割
    #--------------------------------------------------
    prePro3 = DataPreProcess.DataPreProcess()

    # Wine データセットの読み込み
    prePro3.setDataFrameFromCsvFile( "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" )
    
    # 上記URLのWine データセットにはラベルがついてないので, 列名をセット
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

    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( 
        X_input = prePro3.df_.iloc[:, 1:].values,   # iloc : 行、列を番号で指定（先頭が 0）。df_.iloc[:, 1:] = 全行、1~の全列
        y_input = prePro3.df_.iloc[:, 0].values,    #
        ratio_test = 0.3
    )

    # 分割データ（トレーニングデータ、テストデータ）を出力
    print( "トレーニングデータ : \n", X_train )
    print("テストデータ : \n", X_test )
    print("トレーニング用教師データ : \n", y_train )
    print("テスト用教師データ : \n", y_test )

    #--------------------------------------------------
    # Practice 4 : 特徴量のスケーリング
    # 正規化 [normalization], 標準化 [standardization]
    #--------------------------------------------------
    # 正規化
    X_train_norm, X_test_norm \
    = DataPreProcess.DataPreProcess.normalizedTrainTest( X_train, X_test )
    
    # 正規化後のデータを出力
    print( "トレーニングデータ [normalized] :\n", X_train_norm )
    print("テストデータ [normalized] : \n", X_test_norm )

    # 標準化
    X_train_std, X_test_std \
    = DataPreProcess.DataPreProcess.standardizeTrainTest( X_train, X_test )

    # 標準化後のデータを出力
    print( "トレーニングデータ [standardized] :\n", X_train_std )
    print("テストデータ [standardized] : \n", X_test_std )


    #--------------------------------------------------------
    # Practice 5 : 有益な特徴量の選択
    # L1正則化による疎な解（ロジスティクス回帰モデルで検証）
    #--------------------------------------------------------
    logReg = LogisticRegression(
        penalty = 'l1',     # L1正則化
        C = 0.1             # 逆正則化パラメータ
    )

    logReg.fit( X_train_std, y_train )
    
    print( 'Training accuracy:', logReg.score( X_train_std, y_train ) )
    print( 'Test accuracy:', logReg.score( X_test_std, y_test ) )

    print("切片 :",logReg.intercept_ )
    print("重み係数 : \n",logReg.coef_ )

    #----------------------------------------
    # 図の作図
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    
    # 各係数（特徴）の色のリスト
    colors = [
                 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange'
             ]

    # 重み係数のリスト、逆正則化パラメータのリスト（空のリストで初期化）
    weights = [] 
    params = []

    # 逆正則化パラメータの値毎に処理を繰り返す
    for c in numpy.arange(-4., 6.):
        lr = LogisticRegression( penalty = 'l1', C = 10.**c, random_state = 0 )
        lr.fit( X_train_std, y_train )
        weights.append( lr.coef_[1] )
        params.append( 10.**c )

    weights = numpy.array( weights )    # 重み係数を Numpy 配列に変換

    # 各重み係数をプロット
    for column, color in zip( range(weights.shape[1]), colors ):
        # 折れ線グラフ
        plt.plot(
            params, weights[:, column],
            label = prePro3.df_.columns[column + 1],
            color = color
        )
    
    plt.grid()
    plt.axhline( 0, color = 'black', linestyle = '--', linewidth = 3 )
    plt.xlim( [10**(-5), 10**5] )
    plt.ylabel('weight coefficient')
    plt.xlabel('C [Reverse regularization parameter] (log scale)')
    plt.xscale('log')   # x 軸を log スケール化
    plt.legend(loc='lower left')
    """
    ax.legend(
        loc = 'upper center', 
        #bbox_to_anchor = (1.38, 1.03),
        ncol = 1
        #fancybox = True
    )
    """
    plt.tight_layout()

    plt.savefig( 'DataPreProcess_scikit-learn_1.png', dpi=300, bbox_inches = 'tight' )
    plt.show()


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

