# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder          

from sklearn.model_selection import cross_val_score     #
from sklearn.metrics import accuracy_score              # 
from sklearn.metrics import roc_curve                   # ROC曲線
from sklearn.metrics import auc                         # AUC


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC                             # 

from sklearn.model_selection import learning_curve      # 学習曲線用
from sklearn.model_selection import validation_curve    # 検証曲線用
from sklearn.model_selection import GridSearchCV        # 

from sklearn.pipeline import Pipeline

# 自作クラス
import EnsembleModelClassifier
import DataPreProcess
import Plot2D


def main():
    """
    アンサンブル学習.
    多数決方式のアンサンブル分類器と、異なるモデルの組み合わせ
    """
    print("Enter main()")

    # データの読み込み
    # 三品種 (Setosa, Versicolor, Virginica) の特徴量、含まれる特徴量は、Sepal (がく片) と Petal (花びら) の長さと幅。
    iris = datasets.load_iris()         
    print(iris)

    dat_X = iris.data[ 50:, [1, 2] ]    # 
    dat_y = iris.target[50:]            # 

    dat_X, dat_y = DataPreProcess.DataPreProcess.generateCirclesDataSet()
    dat_X, dat_y = DataPreProcess.DataPreProcess.generateMoonsDataSet()
    
    ratio_test = 0.3

    #print(dat_X)
    #print(dat_y)


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
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = dat_X, y_input = dat_y, ratio_test = ratio_test, input_random_state = 1 )

    test_idx = []
    #test_idx = range( 26,50 )

    #
    stdScaler = StandardScaler()
    
    # X_train の平均値と標準偏差を計算
    stdScaler.fit( X_train )

    # 求めた平均値と標準偏差を用いて標準化
    X_train_std = stdScaler.transform( X_train )
    X_test_std  = stdScaler.transform( X_test )

    # 分割したデータを行方向に結合（後で plot データ等で使用する）
    X_combined_std = numpy.vstack( (X_train_std, X_test_std) )  # list:(X_train_std, X_test_std) で指定
    y_combined     = numpy.hstack( (y_train, y_test) )


    #print( "X_train :\n",  X_train )
    #print( "X_test :\n",  X_test )
    #print( "y_train :\n",  y_train )
    #print( "y_test :\n",  y_test )

    #-------------------------------------------
    # 各識別器 classifier の設定
    #-------------------------------------------
    clf1 = LogisticRegression(
               penalty = 'l2', 
               C = 0.001,
               random_state = 0
           )

    clf2 = DecisionTreeClassifier(
               max_depth = 3,
               criterion = 'entropy',
               random_state = 0
           )

    clf3 = KNeighborsClassifier(
               n_neighbors = 3,
               p = 2,
               metric = 'minkowski'
           )

    clf4 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        gamma = 10.0,       # RFBカーネル関数のγ値
        C = 0.1,            # C-SVM の C 値
        random_state = 0,   #
        probability = True  # 学習後の predict_proba method による予想確率を有効にする
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

    pipe4 = Pipeline(
                [                                   
                    ( "sc", StandardScaler() ),  # スケーリング：　変換器のクラス（fit() 関数を持つ）
                    ( "clf", clf4 )              # classifer 4
                ]
            )
    
    #-----------------------------------------------------------
    # アンサンブル識別器 EnsembleLearningClassifier の設定
    #-----------------------------------------------------------
    ensemble_clf1 = EnsembleModelClassifier.EnsembleModelClassifier( 
                        classifiers  = [ pipe1, pipe2, pipe3 ],
                        class_labels = [ 
                                         "Logistic Regression ( penalty = 'l2', C = 0.001 )", 
                                         "Decision Tree ( criterion = 'entropy', max_depth = 3 )", 
                                         "k-NN ( n_neighbors = 3, metric='minkowski' )" 
                                       ]
                    )

        
    ensemble_clf2 = EnsembleModelClassifier.EnsembleModelClassifier( 
                        classifiers = [ pipe1, pipe2, pipe4 ],
                        class_labels = [ 
                                         "Logistic Regression ( penalty = 'l2', C = 0.001 )", 
                                         "Decision Tree ( criterion = 'entropy', max_depth = 3 )", 
                                         "SVM ( kernel = 'rbf', C = 0.1, gamma = 10.0 )" 
                                       ]
                    )
    #ensemble_clf3 = EnsembleModelClassifier.EnsembleModelClassifier( classifiers = [] )
    #ensemble_clf4 = EnsembleModelClassifier.EnsembleModelClassifier( classifiers = [] )

    ensemble_clf1.print( "ensemble_clf1" )
    ensemble_clf2.print( "ensemble_clf2" )

    #------------------------------------------------------------
    # グリッドサーチによる各弱識別器のパラメータのチューニング
    #------------------------------------------------------------
    print( ensemble_clf2.get_params() )

    params2 = {
                #"pipeline-1__clf__C":         [0.001, 0.1, 1, 10, 100.0],   # LogisticRegression の C 値
                #"pipeline-2__clf__max_depth": [1, 2, 3, 4, 5],              # DecisionTree の深さ
                "pipeline-3__clf__C":         [0.001, 0.1, 1, 10, 100.0],
                "pipeline-3__clf__gamma":     [0.001, 0.1, 1, 10, 100.0]
             }

    grid = GridSearchCV(
               estimator = ensemble_clf2,
               param_grid = params2,
               cv = 10,
               scoring = 'roc_auc'
           )
    
    grid.fit( X_train, y_train )

    cv_keys = ('mean_test_score', 'std_test_score','params')
    for r, _ in enumerate( grid.cv_results_['mean_test_score'] ):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r], 
                 grid.cv_results_[cv_keys[1]][r] / 2.0, 
                 grid.cv_results_[cv_keys[2]][r]))
    
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

    """
    グリッドサーチ結果Memo
    SVM
    Best parameters: {'pipeline-3__clf__C': 0.1, 'pipeline-3__clf__gamma': 10}
    Accuracy: 1.00


    """
    #============================================
    # Learning Process
    #===========================================
    # 設定した推定器をトレーニングデータで fitting
    ensemble_clf1.fit( X_train_std, y_train )
    ensemble_clf2.fit( X_train_std, y_train )

    #===========================================
    # 汎化性能の確認
    #===========================================

    # 各種スコア計算時に使用する識別器のリスト ( for 文の in で使用を想定) 
    all_clf = []
    all_clf = ensemble_clf2.classifiers_
    all_clf.append( ensemble_clf2 )
    print( "all_clf :", all_clf )

    # 各種スコア計算時に使用するクラスラベルのリスト ( for 文の in で使用を想定)
    all_clf_labels = []
    all_clf_labels = ensemble_clf2.class_labels_
    #all_clf_labels.append( "Ensemble Model 1 ( LogisticRegression, DecisionTree, k-NN)" )
    all_clf_labels.append( "Ensemble Model 2 ( LogisticRegression, DecisionTree, SVM)" )
    print( "all_clf_labels :", all_clf_labels )

    #---------------------------------------------------------------
    # Ensemble Model のスコア値
    #---------------------------------------------------------------
    """
    # テストデータ X_test でクラスラベルを予想
    y_train_predict_emb1 = ensemble_clf1.predict( X_train_std )
    y_test_predict_emb1 = ensemble_clf1.predict( X_test_std )
    y_train_predict_emb2 = ensemble_clf2.predict( X_train_std )
    y_test_predict_emb2 = ensemble_clf2.predict( X_test_std )

    # 正解率
    accuracy_train_scores1 = accuracy_score( y_train, y_train_predict_emb1 )
    accuracy_test_scores1 = accuracy_score( y_test, y_test_predict_emb1 )
    accuracy_train_scores2 = accuracy_score( y_train, y_train_predict_emb2 )    #
    accuracy_test_scores2 = accuracy_score( y_test, y_test_predict_emb2 )       #

    # AUC
    fpr, tpr, thresholds = roc_curve( y_train, y_train_predict_emb1, pos_label = 1 )
    auc_train_scores1 = auc( fpr, tpr )
    fpr, tpr, thresholds = roc_curve( y_test, y_test_predict_emb1, pos_label = 1 )
    auc_test_scores1 = auc( fpr, tpr )

    fpr, tpr, thresholds = roc_curve( y_train, y_train_predict_emb2, pos_label = 1 )
    auc_train_scores2 = auc( fpr, tpr )
    fpr, tpr, thresholds = roc_curve( y_test, y_test_predict_emb2, pos_label = 1 )
    auc_test_scores2 = auc( fpr, tpr )

    print( "[Ensemble Model のスコア値 cv = None]")
    print( "Accuracy <train data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( accuracy_train_scores1.mean(), accuracy_train_scores1.std(), "Ensemble Model 1") )
    print( "Accuracy <test data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( accuracy_test_scores1.mean(), accuracy_test_scores1.std(), "Ensemble Model 1") )
    print( "Accuracy <train data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( accuracy_train_scores2.mean(), accuracy_train_scores2.std(), "Ensemble Model 2") )
    print( "Accuracy <test data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( accuracy_test_scores2.mean(), accuracy_test_scores2.std(), "Ensemble Model 2") )
    
    print( "AUC <train data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( auc_train_scores1.mean(), auc_train_scores1.std(), "Ensemble Model 1") )
    print( "AUC <test data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( auc_test_scores1.mean(), auc_test_scores1.std(), "Ensemble Model 1") )
    print( "AUC <train data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( auc_train_scores2.mean(), auc_train_scores2.std(), "Ensemble Model 2") )
    print( "AUC <test data> : %0.2f (+/- %0.2f) cv=None [%s]" % ( auc_test_scores2.mean(), auc_test_scores2.std(), "Ensemble Model 2") )

    print( "\n")
    """
    #-------------------------------------------
    # 正解率, 誤識率
    #-------------------------------------------
    # k-fold CV を行い, cross_val_score( scoring = 'accuracy' ) で 正解率を算出
    print( "[Accuracy]")
    # train data
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_train,
                     y = y_train,
                     cv = 10,
                     scoring = 'accuracy'    # 正解率
                 )
        print( "Accuracy <train data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )    
    
    # test data
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_test,
                     y = y_test,
                     cv = 10,
                     scoring = 'accuracy'    # 正解率
                 )
        print( "Accuracy <test data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )    

    
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
        print( "AUC <train data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )

    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_test,
                     y = y_test,
                     cv = 10,
                     scoring = 'roc_auc'    # AUC
                 )
        print( "AUC <test data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )

    #-------------------------------------------
    # 識別境界
    #-------------------------------------------
    plt.clf()

    plt.subplot( 2, 2, 1 )
    Plot2D.Plot2D.drawDiscriminantRegions( X_combined_std, y_combined, classifier = all_clf[0] )
    plt.title( all_clf_labels[0] )
    plt.xlabel( "Sepal width [standardized]" )
    plt.ylabel( "Petal length [standardized]" )
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.subplot( 2, 2, 2 )
    Plot2D.Plot2D.drawDiscriminantRegions( X_combined_std, y_combined, classifier =  all_clf[1] )
    plt.title( all_clf_labels[1] )
    plt.xlabel( "Sepal width [standardized]" )
    plt.ylabel( "Petal length [standardized]" )
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.subplot( 2, 2, 3 )
    Plot2D.Plot2D.drawDiscriminantRegions( X_combined_std, y_combined, classifier =  all_clf[2] )
    plt.title( all_clf_labels[2] )
    plt.xlabel( "Sepal width [standardized]" )
    plt.ylabel( "Petal length [standardized]" )
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.subplot( 2, 2, 4 )
    Plot2D.Plot2D.drawDiscriminantRegions( X_combined_std, y_combined, classifier = all_clf[3] )
    plt.title( all_clf_labels[3] )
    plt.xlabel( "Sepal width [standardized]" )
    plt.ylabel( "Petal length [standardized]" )
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.savefig("./EnsembleLearning_scikit-learn_2.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    #-------------------------------------------
    # 学習曲線
    #-------------------------------------------
    plt.clf()

    for (idx, clf) in zip( range(1,5),  all_clf ):
        #print(idx)
        #print(clf)

        train_sizes, train_scores, test_scores \
        = learning_curve(
              estimator = clf,    # 推定器 
              X = X_train_std,                              # 
              y = y_train,                                  # 
              train_sizes = numpy.linspace(0.1, 1.0, 10),   # トレードオフサンプルの絶対数 or 相対数
                                                            # トレーニングデータサイズに応じた, 等間隔の10 個の相対的な値を設定
              cv = 10                                       # 交差検証の回数（分割数）
        )

        # 平均値、分散値を算出
        train_means = numpy.mean( train_scores, axis = 1 )   # axis = 1 : 行方向
        train_stds = numpy.std( train_scores, axis = 1 )
        test_means = numpy.mean( test_scores, axis = 1 )
        test_stds = numpy.std( test_scores, axis = 1 )

        # idx 番目の plot
        plt.subplot( 2, 2, idx )
        Plot2D.Plot2D.drawLearningCurve(
            train_sizes = train_sizes,
            train_means = train_means,
            train_stds = train_stds,
            test_means = test_means,
            test_stds = test_stds
        )
        plt.title( "Learning Curve \n" + all_clf_labels[idx-1] )
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.ylim( [0.25, 1.01] )
        plt.tight_layout()
        
    
    plt.savefig("./EnsembleLearning_scikit-learn_3.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

  
    #-------------------------------------------
    # ROC 曲線
    #-------------------------------------------
    plt.clf()

    Plot2D.Plot2D.drawROCCurveFromClassifiers( 
        classifilers = all_clf, 
        class_labels = all_clf_labels, 
        X_train = X_train_std, y_train = y_train,
        X_test = X_test_std, y_test = y_test
    )

    
    plt.savefig("./EnsembleLearning_scikit-learn_4.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()    
    
        

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()