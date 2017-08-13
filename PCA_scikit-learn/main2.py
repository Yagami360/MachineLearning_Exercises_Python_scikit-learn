# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    print("Enter main()")
    #=============================================================================================
    # 主成分分析 [PCA : Principal Component Analysis] による教師なしデータの次元削除、特徴抽出
    # scikit-learn ライブラリでの主成分分析使用
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
            'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
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
    
    #========================================================================
    # Learning Process (scikit-learn のPCAクラスを使用)＆ロジスティクス回帰
    #========================================================================
    # PCA で次元削除
    pca = PCA( n_components = 2 )   # n_components : 主成分数（PC1,PC2）
    
    X_train_pca = pca.fit_transform( X_train_std )
    X_test_pca = pca.transform( X_test_std )

    # ロジスティクス回帰
    logReg = LogisticRegression()
    logReg = logReg.fit( X_train_pca, y_train )     # 次元削除したトレーニングデータ X_train_pca で識別

    #====================================================
    #   汎化性能の評価
    #====================================================
    #--------------------------------------------------------
    # 13 次元 → 2 次元に次元削除した主成分空間での散布図
    #--------------------------------------------------------
    # 学習データ
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_train_pca, dat_y = y_train,
        classifier = logReg
    )
    plt.title("Idefication Result - Learning data \n Logistic Regression (dimension is deleted by PCA)")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./PCA_scikit-learn_4.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    # テストデータ
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_test_pca, dat_y = y_test,
        classifier = logReg
    )

    plt.title("Idefication Result - test data \n Logistic Regression (dimension is deleted by PCA)")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./PCA_scikit-learn_5.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()
