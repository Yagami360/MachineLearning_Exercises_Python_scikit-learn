# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス

from sklearn.model_selection import learning_curve      # 学習曲線、検証曲線用

from sklearn.pipeline import Pipeline

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    """
    機械学習パイプラインによる、機械学習処理フロー（scikit-learn ライブラリの Pipeline クラスを使用）
    学習曲線, 検証曲線よるモデルの汎化性能の評価
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
                                  ( "clf", LogisticRegression(penalty='l2', random_state=0) ) # ロジスティクス回帰（L2正則化）：推定器のクラス（predict()関数を持つ）
                              ]
                  )

    
    # パイプラインに設定した変換器の fit() 関数を実行
    #pipe_logReg.fit( X_train, y_train )

    # 
    #print( "Test Accuracy: %.3f" % pipe_logReg.score( X_test, y_test ) )

    #============================================
    # Learning Process
    #===========================================
    # パイプラインに設定した推定器の predict() 実行
    #y_predict = pipe_logReg.predict(X_test)
    #print("predict : ", y_predict )
    

    #===========================================
    # 汎化性能の確認
    #===========================================
    # learning_curve() 関数で"交差検証"による正解率を算出
    train_sizes, train_scores, test_scores \
    = learning_curve(
          estimator = pipe_logReg,                      # 推定器 : Pipeline に設定しているロジスティクス回帰
          X = X_train,                                  # 
          y = y_train,                                  # 
          train_sizes = numpy.linspace(0.1, 1.0, 10),   # トレードオフサンプルの絶対数 or 相対数
                                                        # トレーニングデータサイズに応じた, 等間隔の10 個の相対的な値を設定
          cv = 10,                                      # 交差検証の回数（分割数）
          n_jobs = -1                                   # 全てのCPUで並列処理
      )

    # 平均値、分散値を算出
    train_means = numpy.mean( train_scores, axis = 1 )   # axis = 1 : 行方向
    train_stds = numpy.std( train_scores, axis = 1 )
    test_means = numpy.mean( test_scores, axis = 1 )
    test_stds = numpy.std( test_scores, axis = 1 )

    #
    print( "train_sizes : \n" , train_sizes )           # トレーニングデータサイズに応じた, 等間隔の10 個の相対的な値のリスト
    print( "train_scores : \n" , train_scores )
    print( "test_scores : \n" , test_scores )
    print( "train_means : \n" , train_means )
    print( "train_stds : \n" , train_stds )
    print( "test_means : \n" , test_means )
    print( "test_stds : \n" , test_stds )

    #-------------------------------------------
    # 学習曲線を描写
    #-------------------------------------------
    Plot2D.Plot2D.drawLearningCurve(
        train_sizes = train_sizes,
        train_means = train_means,
        train_stds = train_stds,
        test_means = test_means,
        test_stds = test_stds
    )
    plt.title( "Learning Curve" )
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim( [0.8, 1.01] )
    plt.tight_layout()

    plt.savefig("./MachineLearningPipeline_scikit-learn_1.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    #-------------------------------------------
    # 検証曲線を描写
    #-------------------------------------------


    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()