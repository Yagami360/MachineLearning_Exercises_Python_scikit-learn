# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.model_selection import StratifiedKFold     #
from sklearn.model_selection import cross_val_score     #

from sklearn.pipeline import Pipeline

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    """
    機械学習パイプラインによる、機械学習処理フロー（scikit-learn ライブラリの Pipeline クラスを使用）
    クロス・バディゲーションによる汎化性能の確認
    """
    print("Enter main()")
    
    # データの読み込み
    prePro = DataPreProcess.DataPreProcess()
    prePro.setDataFrameFromCsvFile(
        "https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data"
    )
    #prePro.print( "Breast Cancer Wisconsin dataset" )
    
    dat_X = prePro.df_.loc[:, 2:].values
    dat_y = prePro.df_.loc[:, 1].values

    #===========================================
    # 前処理 [PreProcessing]
    #===========================================
    # 欠損データへの対応
    #prePro.meanImputationNaN()

    # ラベルデータをエンコード
    prePro.encodeClassLabelByLabelEncoder( colum = 1 )
    prePro.print( "Breast Cancer Wisconsin dataset" )

    # データをトレードオフデータとテストデータに分割
    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = dat_X, y_input = dat_y, ratio_test = 0.2 )

    #-------------------------------------------
    # Pipeline の設定
    #-------------------------------------------
    # パイプラインに各変換器、推定器を設定
    pipe_logReg = Pipeline(
                      steps = [                                           # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定
                                  ( "scl", StandardScaler() ),            # スケーリング：　変換器のクラス（fit() 関数を持つ）
                                  ( "pca", PCA( n_components=2 ) ),       # PCA でデータの次元削除
                                  ( "clf", LogisticRegression( random_state=1 ) ) # ロジスティクス回帰：推定器のクラス（preddict()関数を持つ）
                              ]
                  )

    #
    """ 
    pipe_logReg.set_params(
        [
            ( "scl", StandardScaler() ),            # スケーリング：　変換器のクラス（fit() 関数を持つ）
            ( "pca", PCA( n_components=2 ) ),       # PCA でデータの次元削除
            ( "clf", LogisticRegression( random_state=1 ) ) # ロジスティクス回帰：推定器のクラス（preddict()関数を持つ）
        ]    
    )
    """

    # パイプラインに設定した変換器の fit() 関数を実行
    pipe_logReg.fit( X_train, y_train )

    # 
    print( "Test Accuracy: %.3f" % pipe_logReg.score( X_test, y_test ) )

    #============================================
    # Learning Process
    #===========================================
    # パイプラインに設定した推定器の predict() 実行
    y_predict = pipe_logReg.predict(X_test)
    print("predict : ", y_predict )
    
    # pipeline オブジェクトの内容確認
    #print( "pipe_logReg.get_params() : \n", pipe_logReg.get_params( deep = True ) )
    #print( "pipe_logReg.get_params() : \n", pipe_logReg.get_params( deep = False ) )

    #===========================================
    # 汎化性能の確認
    #===========================================
    
    #-------------------------------------------
    # クロスバディゲーション : CV
    #-------------------------------------------
    scores = cross_val_score(
                estimator = pipe_logReg,  # 推定器 [estimator]
                X = X_train,              #
                y = y_train,              #
                cv = 10,                  # 交差検証の回数（分割数）
                n_jobs = -1               # 全てのCPUで並列処理
             )

    print( 'CV accuracy scores: %s' % scores )
    print( 'CV accuracy: %.3f +/- %.3f' % ( numpy.mean(scores), numpy.std(scores) ) )


    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()