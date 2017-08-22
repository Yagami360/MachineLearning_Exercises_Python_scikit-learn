# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.svm import SVC                             # 
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold     #

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve                   # ROC曲線
from sklearn.metrics import auc                         # AUC

from scipy import interp                                # （AUCを計算するための）補間処理

from sklearn.pipeline import Pipeline

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    """
    機械学習パイプラインによる、機械学習処理フロー（scikit-learn ライブラリの Pipeline クラスを使用）
    ROC 曲線によるモデルの汎化能力の評価
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

    prePro.encodeClassLabelByLabelEncoder( colum = 0 )
    prePro.print( "Breast Cancer Wisconsin dataset" )

    # データをトレードオフデータとテストデータに分割
    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = dat_X, y_input = dat_y, ratio_test = 0.2 )

    #-------------------------------------------
    # Pipeline の設定
    #-------------------------------------------
    # パイプラインに各変換器、推定器を設定
    pipe_logReg = Pipeline(
                      steps = [                                     # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定
                                  ( "scl", StandardScaler() ),      # 正規化 : 変換器のクラス（fit() 関数を持つ）
                                  ('pca', PCA(n_components=2)),     # PCA で２次元に削除（特徴抽出）
                                  ('clf', LogisticRegression(penalty='l2', random_state=0, C=100.0) )  # ロジスティクス回帰（L2正則化） 
                                                                                                       # 推定器のクラス（predict()関数を持つ）
                              ]
                  )

    # ? 使用するデータ（特徴量）
    #X_train2 = X_train[:, [4, 14]]

    #-------------------------------------------
    # クロスバリデーションの設定
    #-------------------------------------------
    # クロスバディゲーションの回数毎の ROC 曲線を描写するため、
    # クラスのオブジェクト作成（分割数：３）を作成.
    cv = StratifiedKFold( n_splits=3, random_state=1 )

    # クラスのオブジェクトをイテレータ化するために split() して list 化
    list_cv = list( cv.split( X_train, y_train ) )

    print( "StratifiedKFold() : \n", cv )
    print( "list( StratifiedKFold().split() ) : \n", list_cv )

    #-------------------------------------------
    # 
    #-------------------------------------------
    # パイプラインに設定した変換器の fit() 関数を実行
    #pipe_logReg.fit( X_train, y_train )

    # 予想値
    #y_predict =pipe_logReg.predict( X_test )
    
    # pipeline オブジェクトの内容確認
    print( "Pipeline.get_params( deep = True ) : \n", pipe_logReg.get_params( deep = True ) )
    print( "Pipeline.get_params( deep = False ) : \n", pipe_logReg.get_params( deep = False ) )

    #print( "Pipeline.predict( X_test ) : \n", y_predict )
    #print( "Pipeline.predict( X_test )[0] : \n", y_predict[0] )
    #print( "Pipeline.predict( X_test )[1] : \n", y_predict[1] )
    #print( "Pipeline.predict( X_test )[2] : \n", y_predict[2] )
    #print( "Test Accuracy: %.3f" % pipe_csvm.score( X_test, y_test ) )
    
    #------------------------------------
    # ROC 曲線
    #------------------------------------
    





    plt.savefig("./MachineLearningPipeline_scikit-learn_5.png", dpi = 300, bbox_inches = 'tight' )
    #plt.show()

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()