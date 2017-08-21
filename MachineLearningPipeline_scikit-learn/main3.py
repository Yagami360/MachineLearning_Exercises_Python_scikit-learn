# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.svm import SVC                             # 
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.model_selection import GridSearchCV        # 

from sklearn.pipeline import Pipeline

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    """
    機械学習パイプラインによる、機械学習処理フロー（scikit-learn ライブラリの Pipeline クラスを使用）
    グリッドサーチによるハイパーパラメータのチューニング
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
    pipe_csvm = Pipeline(
                      steps = [                                     # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定
                                  ( "scl", StandardScaler() ),      # 正規化 : 変換器のクラス（fit() 関数を持つ）
                                  ('clf', SVC( random_state=1 ) )   # C-SVM : 推定器のクラス（predict()関数を持つ）
                              ]
                  )

    
    # パイプラインに設定した変換器の fit() 関数を実行
    #pipe_csvm.fit( X_train, y_train )

    # 
    # pipeline オブジェクトの内容確認
    print( "Pipeline.get_params() : \n", pipe_csvm.get_params( deep = True ) )
    print( "Pipeline.get_params() : \n", pipe_csvm.get_params( deep = False ) )

    #print( "Test Accuracy: %.3f" % pipe_csvm.score( X_test, y_test ) )

    
    #==============================
    # grid search
    #==============================
    # グリッドサーチの対象パラメータ : 今の場合 C=SVM の正規化パラメータ C 値とガンマ値
    param_range_C = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_range_gamma = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # グリッドサーチでチューニングしたいモデルとそのパラメータ : ディクショナリ（辞書）のリストで指定
    param_grid = [
        { 'clf__C': param_range_C, 'clf__kernel': ['linear'] },                                # liner C-SVM
        { 'clf__C': param_range_C, 'clf__gamma': param_range_gamma, 'clf__kernel': ['rbf'] }   # RBF-kernel C-SVM
    ]

    # グリッドサーチを行う,GridSearchCV クラスのオブジェクト作成
    gs = GridSearchCV(
            estimator = pipe_csvm,      # 推定器
            param_grid = param_grid,    # グリッドサーチの対象パラメータ
            scoring = 'accuracy',       # 
            cv = 10,                    # クロスバディゲーションの回数
            n_jobs = -1                 # 全てのCPUで並列処理
         )
    # グリッドサーチを行う
    gs = gs.fit( X_train, y_train )


    # グリッドサーチの結果を print
    print( "sklearn.model_selection.GridSearchCV.best_score_ : \n", gs.best_score_ )        # 指定したモデルの内, 最もよいスコアを出したモデルのスコア
    print( "sklearn.model_selection.GridSearchCV.best_params_ : \n", gs.best_params_ )      # 最もよいスコアを出したモデルのパラメータ
    #print( "sklearn.model_selection.GridSearchCV.grid_scores_ : \n",gs.grid_scores_ )       # 全ての情報
    
    # 最もよいスコアを出したモデルを抽出し, テストデータを評価
    clf = gs.best_estimator_
    clf.fit( X_train, y_train )     # 抽出したモデルをトレーニングデータで学習
    print('sklearn.model_selection.GridSearchCV.best_estimator_ in Test accuracy: %.3f' % clf.score( X_test, y_test ) )     # 最もよいスコアを出したモデルでのテストデータ

    #-----------------------------------------------
    # グリッドサーチのためのヒートマップ図の plot
    #-----------------------------------------------
    # 再設定：RBF-kernel SVM
    param_grid = [
        { 'clf__C': param_range_C, 'clf__gamma': param_range_gamma, 'clf__kernel': ['rbf'] }    # RBF-kernel C-SVM
    ]

    # グリッドサーチを行う,GridSearchCV クラスのオブジェクト作成
    gs = GridSearchCV(
            estimator = pipe_csvm,      # 推定器
            param_grid = param_grid,    # グリッドサーチの対象パラメータ
            scoring = 'accuracy',       # 
            cv = 10,                    # クロスバディゲーションの回数
            n_jobs = -1                 # 全てのCPUで並列処理
         )

    # グリッドサーチを行う
    gs = gs.fit( X_train, y_train )

    # grid_scores_ 属性から正解率を抽出
    gs_params = []
    gs_mean_scores = []
    gs_scores = []

    for parames, mean_score, scores in gs.grid_scores_:
        gs_params.append( parames )
        gs_mean_scores.append( mean_score )
        gs_scores.append( scores )
    
    
    gs_mean_scores = numpy.reshape( gs_mean_scores , ( len(param_range_C), len(param_range_gamma) ) )
    #gs_scores = numpy.reshape( gs_scores , (8,8) )

    print( "sklearn.model_selection.GridSearchCV.grid_scores_.parmes : \n", gs_params )
    print( "sklearn.model_selection.GridSearchCV.grid_scores_.mean_scores : \n", gs_mean_scores )
    print( "sklearn.model_selection.GridSearchCV.grid_scores_.scores : \n", gs_scores )

    # ヒートマップのためのデータ
    heatmap_Z = gs_mean_scores
    heatmap_x = param_range_gamma
    heatmap_y = param_range_C
    
    # ヒートマップを作図
    Plot2D.Plot2D.drawHeatMapFromGridSearch(
        dat_Z = heatmap_Z,        # ヒートマップの値 : RBF-kernel SVM での正解率
        dat_x = heatmap_x,        # x 軸の目盛り
        dat_y = heatmap_y         # y 軸の目盛り
    )
    
    plt.title("Heat Map (Grid Serch) \n values : Accuracy , classifiler : RBF-kernel SVM")
    plt.ylabel( "C : RBF-kernel SVM parametor" )
    plt.xlabel( "gamma : RBF-kernel parametor" )

    plt.savefig("./MachineLearningPipeline_scikit-learn_3.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()


    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()