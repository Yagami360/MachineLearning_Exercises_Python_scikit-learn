# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder          
from sklearn.model_selection import cross_val_score     #

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.pipeline import Pipeline

# 自作クラス
import EnsembleLearningClassifier
import DataPreProcess
import Plot2D


def main():
    """
    アンサンブル学習.
    多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ
    """
    print("Enter main()")

    # データの読み込み
    iris = datasets.load_iris()

    dat_X = iris.data[ 50:, [1, 2] ]
    dat_y = iris.target[50:]


    """
    # 重み付き多数決 : y_predict = arg max∑w*x(classifiler = i)
    # numpy.bincount() : 引数 x における各値の出現に対し, 引数 weights で重み付けする. 
    weight_result = numpy.bincount( [0, 0, 1], weights = [0.2, 0.2, 0.6] )
    # numpy.argmax() : 引数 a の値の内で最大値のインデックス値
    argmax_result = numpy.argmax( a = weight_result )
    print( "numpy.bincount() : \n", weight_result )
    print( "numpy.argmax() : \n", argmax_result )
    """

    #===========================================
    # 前処理 [PreProcessing]
    #===========================================
    # 欠損データへの対応
    #prePro.meanImputationNaN()

    # ラベルデータをエンコード
    #prePro.encodeClassLabelByLabelEncoder( colum = 1 )
    #prePro.print( "" )
    encoder = LabelEncoder()
    dat_y = encoder.fit_transform( dat_y )

    # データをトレードオフデータとテストデータに分割
    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = dat_X, y_input = dat_y, ratio_test = 0.5, input_random_state = 1 )

    print( "X_train :\n",  X_train )
    print( "X_test :\n",  X_test )
    print( "y_train :\n",  y_train )
    print( "y_test :\n",  y_test )

    #-------------------------------------------
    # 各識別器 classifier の設定
    #-------------------------------------------
    clf1 = LogisticRegression(
               penalty = 'l2', 
               C = 0.001,
               random_state = 0
           )

    clf2 = DecisionTreeClassifier(
               max_depth = 1,
               criterion = 'entropy',
               random_state = 0
           )

    clf3 = KNeighborsClassifier(
               n_neighbors = 1,
               p = 2,
               metric = 'minkowski'
           )

    #-------------------------------------------
    # 各 Pipeline の設定
    #-------------------------------------------
    # パイプラインに各変換器、推定器を設定
    # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定
    pipe1 = Pipeline(
                [                                   
                    ( "sc", StandardScaler() ),  # スケーリング：　変換器のクラス（fit() 関数を持つ）
                    ( "clf", clf1 )              # classifer 1
                ]
            )

    pipe2 = Pipeline(
                [                                   
                    ( "sc", StandardScaler() ),  # スケーリング：　変換器のクラス（fit() 関数を持つ）
                    ( "clf", clf2 )              # classifer 2
                ]
            )

    pipe3 = Pipeline(
                [                                   
                    ( "sc", StandardScaler() ),  # スケーリング：　変換器のクラス（fit() 関数を持つ）
                    ( "clf", clf3 )              # classifer 3
                ]
            )
    
    #-----------------------------------------------------------
    # アンサンブル識別器 EnsembleLearningClassifier の設定
    #-----------------------------------------------------------
    ensemble_clf1 = EnsembleLearningClassifier.EnsembleLearningClassifier( 
                        classifiers = [ pipe1, clf2, pipe3 ],
                        class_labels = [ "Logistic Regression", "Decision Tree", "k-NN" ]
                    )

        
    #ensemble_clf2 = EnsembleLearningClassifier.EnsembleLearningClassifier( classifiers = [] )
    #ensemble_clf3 = EnsembleLearningClassifier.EnsembleLearningClassifier( classifiers = [] )
    #ensemble_clf4 = EnsembleLearningClassifier.EnsembleLearningClassifier( classifiers = [] )

    ensemble_clf1.print( "ensemble_clf1" )

    # 各種スコア計算時に使用する識別器のリスト ( for 文の in で使用を想定) 
    all_clf = []
    all_clf = ensemble_clf1.get_classiflers()
    all_clf.append( ensemble_clf1 )
    print( "all_clf :", all_clf )

    # 各種スコア計算時に使用するクラスラベルのリスト ( for 文の in で使用を想定)
    all_clf_labels = []
    all_clf_labels = ensemble_clf1.get_class_labels()
    all_clf_labels.append( "Ensemble Model 1" )
    print( "all_clf_labels :", all_clf_labels )

    #============================================
    # Learning Process
    #===========================================
    # 設定した推定器をトレーニングデータで fitting
    #ensemble_clf1.fit( X_train, y_train )

    #===========================================
    # 汎化性能の確認
    #===========================================
    # テストデータ X_test でクラスラベルを予想
    #y_predict = ensemble_clf1.predict( X_test )
    #print( "ensemble_clf1.predict() : " , y_predict )


    #-------------------------------------------
    # 正解率, 誤識率
    #-------------------------------------------
    # k-fold CV を行い, cross_val_score( scoring = 'accuracy' ) で 正解率を算出
    print( "[Accuracy]")
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_train,
                     y = y_train,
                     cv = 10,
                     scoring = 'accuracy'    # 正解率
                 )
        print( "Accuracy : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )    

    #scores_accuracy = cross_val_score( estimator = ensemble_clf1
    #-------------------------------------------
    # 識別境界
    #-------------------------------------------


    #-------------------------------------------
    # AUC 値
    #-------------------------------------------
    # k-fold CV を行い, cross_val_score( scoring = 'roc_auc' ) で AUC を算出
    print( "[AUC]")
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_train,
                     y = y_train,
                     cv = 10,
                     scoring = 'roc_auc'    # AUC
                 )
        print( "AUC : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )

    #-------------------------------------------
    # ROC 曲線
    #-------------------------------------------
    
    
    
    #-------------------------------------------
    # グリッドサーチ with ヒートマップ
    #-------------------------------------------


    """
    plt.subplot(2,2,1)
    ensemble_clf1.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,2)
    ensemble_clf2.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,3)
    ensemble_clf3.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,4)
    ensemble_clf4.plotEnsenbleErrorAndBaseError()

    plt.savefig("./EnsembleLearning_scikit-learn_2.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()
    """

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()