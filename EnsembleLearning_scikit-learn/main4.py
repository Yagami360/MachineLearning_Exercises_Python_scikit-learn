# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets

from sklearn.preprocessing import LabelEncoder          
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                             # 
from sklearn.ensemble import BaggingClassifier          # バギング
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost
from sklearn.ensemble import RandomForestClassifier     # 

from sklearn.metrics import accuracy_score              # 正解率の算出
from sklearn.metrics import roc_curve                   # ROC曲線
from sklearn.metrics import auc                         # AUC
from sklearn.model_selection import cross_val_score     # k-flod CV での各種スコア
from sklearn.model_selection import learning_curve      # 学習曲線用
from sklearn.model_selection import validation_curve    # 検証曲線用
from sklearn.model_selection import GridSearchCV        # グリッドサーチ

from sklearn.pipeline import Pipeline

# 自作クラス
import EnsembleModelClassifier
import DataPreProcess
import Plot2D

def main():
    """
    アンサンブル学習.
    アダブースト
    """
    print("Enter main()")

    # データの読み込み
    """
    # アヤメデータ
    # 三品種 (Setosa, Versicolor, Virginica) の特徴量、含まれる特徴量は、Sepal (がく片) と Petal (花びら) の長さと幅。
    #iris = datasets.load_iris()         
    #print(iris)

    #X_features = iris.data[ 50:, [1, 2] ]    # 
    #y_labels = iris.target[50:]            # 
    """

    # ワインデータセット
    prePro = DataPreProcess.DataPreProcess()
    prePro.setDataFrameFromCsvFile(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    )
    
    prePro.setColumns( 
        ['Class label',
         'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    )

    prePro.print("Breast Cancer Wisconsin dataset")
    
    # 使用する特徴量と教師データ
    # Class label が 1 のサンプルを除外し、ラベルが 2,3 のみのデータを使用
    prePro.df_ = prePro.df_[ prePro.df_['Class label'] != 1 ]
    X_features = prePro.df_[ ['Alcohol', "Hue" ] ].values       # 特徴行列 : 2つの特徴量×サンプル数
    y_labels = prePro.df_[ ['Class label'] ].values             # クラスラベル : 2 or 3

    #X_features, y_labels = DataPreProcess.DataPreProcess.generateCirclesDataSet()
    #X_features, y_labels = DataPreProcess.DataPreProcess.generateMoonsDataSet()

    # 渦巻きデータ
    """
    prePro = DataPreProcess.DataPreProcess()
    prePro.setDataFrameFromCsvFile( "naruto.csv" )
    prePro.setColumns( ["x","y","class labels"] )

    prePro.print( "渦巻きデータ ")

    X_features = prePro.df_[ ["x", "y" ] ].values
    y_labels = prePro.df_[ ["class labels"] ].values
    """

    #print( X_features )

    ratio_test = 0.4

    #===========================================
    # 前処理 [PreProcessing]
    #===========================================
    # 欠損データへの対応
    #prePro.meanImputationNaN()

    # ラベルデータをエンコード
    #prePro.encodeClassLabelByLabelEncoder( colum = 1 )
    #prePro.print( "" )
    encoder = LabelEncoder()
    y_labels = encoder.fit_transform( y_labels )

    # データをトレードオフデータとテストデータに分割
    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = ratio_test, input_random_state = 1 )
    
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
    # モデルの生成
    #-------------------------------------------
    # 決定木の生成
    decition_tree = DecisionTreeClassifier(
                        criterion = 'entropy',       # 不純度として, 交差エントロピー
                        max_depth = None,               # None : If None, then nodes are expanded until all leaves are pure 
                                                     # or until all leaves contain less than min_samples_split samples.(default=None)
                        random_state = 0
                    )
    # k-NN
    kNN = KNeighborsClassifier(
               n_neighbors = 3,
               p = 2,
               metric = 'minkowski'
           )

    # SVM
    svm = SVC( 
            kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
            gamma = 10.0,       # RFBカーネル関数のγ値
            C = 0.1,            # C-SVM の C 値
            random_state = 0,   #
            probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )

    # LogisticRegression
    logReg = LogisticRegression(
                penalty = 'l2', 
                C = 0.001,
                random_state = 0
             )

    # バギングの生成
    bagging = BaggingClassifier(
                  base_estimator = decition_tree,   # 弱識別器をして決定木を設定
                  n_estimators = 501,               # バギングを構成する弱識別器の数
                  max_samples = 1.0,                # The number of samples to draw from X to train each base estimator.
                                                    # If float, then draw max_samples * X.shape[0] samples.
                                                    # base_estimator に設定した弱識別器の内, 使用するサンプルの割合
                                                    # 
                  max_features = 1.0,               # The number of features to draw from X to train each base estimator.
                                                    # If float, then draw max_features * X.shape[1] features.
                  bootstrap = True,                 # ブートストラップサンプリングを行う 
                  bootstrap_features = False,       #
                  n_jobs = -1, 
                  random_state = 0
              )
    
    # AdaBoost
    ada = AdaBoostClassifier(
              base_estimator = decition_tree,       # 弱識別器をして決定木を設定
              n_estimators = 501,                   # バギングを構成する弱識別器の数 
              learning_rate = 0.1,                  # 
              random_state = 0                      #
          )

    # Random Forest
    forest = RandomForestClassifier(
                criterion = "gini",     # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 501,     # 弱識別器（決定木）の数
                n_jobs = -1,            # The number of jobs to run in parallel for both fit and predict ( -1 : 全てのCPUコアで並列計算)
                random_state = 1,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    
    #-------------------------------------------
    # 各 Pipeline の設定
    #-------------------------------------------
    # パイプラインに各変換器、推定器を設定
    # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定

    #-----------------------------------------------------------
    # アンサンブル識別器 EnsembleLearningClassifier の設定
    #-----------------------------------------------------------
    ensemble_clf1 = EnsembleModelClassifier.EnsembleModelClassifier( 
                        classifiers  = [ bagging, ada, forest, decition_tree, logReg, kNN, svm ],
                        class_labels = [ 
                                         "Bagging ( base_estimator = decition_tree, n_estimators = 501 )" ,
                                         "AdaBoost (base_estimator = decition_tree, n_estimators = 501 )"
                                         "Random Forest ( criterion = 'gini', n_estimators = 501)"
                                         "Decision Tree ( criterion = 'entropy' )",                                         
                                         "Logistic Regression( penalty = 'l2', C = 0.001 )",
                                         "k-NN ( n_neighbors = 3, metric='minkowski' )",
                                         "SVM ( kernel = 'rbf', C = 0.1, gamma = 1.0 )"
                                       ]
                    )

    #-------------------------------------------
    # 全識別器のリストの設定
    #-------------------------------------------
    # 各種スコア計算時に使用する識別器のリスト ( for 文の in で使用を想定) 
    #all_clf = [ bagging, ada, forest, decition_tree, svm, ensemble_clf1 ]
    #all_clf = [ bagging, ada, forest, decition_tree, kNN, svm, ensemble_clf1 ]
    all_clf = [ decition_tree, bagging, ada, forest ]
    print( "all_clf :", all_clf )

    # 各種スコア計算時に使用する識別器のラベルのリスト ( for 文の in で使用を想定)
    all_clf_labels = [ 
                        "Decision Tree ( criterion = 'entropy' )",
                        "Bagging ( base_estimator = decition_tree, n_estimators = 101 )",
                        "AdaBoost (base_estimator = decition_tree, n_estimators = 501 )",
                        "RamdomForest (base_estimator = decition_tree, n_estimators = 501 )"
                        #"Logistic Regression( penalty = 'l2', C = 0.001 )",
                        #"k-NN ( n_neighbors = 3, metric='minkowski' )",
                        #"SVM ( kernel = 'rbf', C = 0.1, gamma = 10.0 )",
                        #"Ensemble Model 2 ( Bagging, AdaBoost, RandamForest, Decision Tree, LogisticRegression, k-NN, SVM )"
                     ]

    print( "all_clf_labels :", all_clf_labels )

    #============================================
    # Learning Process
    #===========================================
    # 設定した推定器をトレーニングデータで fitting
    decition_tree = decition_tree.fit( X_train_std, y_train )
    logReg = logReg.fit( X_train_std, y_train )
    kNN = kNN.fit( X_train_std, y_train )
    svm = svm.fit( X_train_std, y_train )
    bagging = bagging.fit( X_train_std, y_train )
    ada = ada.fit( X_train_std, y_train )    
    forest = forest.fit( X_train_std, y_train )
    ensemble_clf1.fit( X_train_std, y_train )

    #print( "decition_tree : ", decition_tree.tree_.max_depth  )
    #print( "bagging : ", bagging )

    #===========================================
    # 汎化性能の確認
    #===========================================

    #-------------------------------------------
    # 正解率, 誤識率
    #-------------------------------------------
    # k-fold CV を行い, cross_val_score( scoring = 'accuracy' ) で 正解率を算出
    print( "[Accuracy]")
    # train data
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_train_std,
                     y = y_train,
                     cv = 10,
                     n_jobs = -1,
                     scoring = 'accuracy'    # 正解率
                 )
        print( "Accuracy <train data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )    
    
    # test data
    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_test_std,
                     y = y_test,
                     cv = 10,
                     n_jobs = -1,
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
                     X = X_train_std,
                     y = y_train,
                     cv = 10,
                     n_jobs = -1,
                     scoring = 'roc_auc'    # AUC
                 )
        print( "AUC <train data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )

    for clf, label in zip( all_clf, all_clf_labels ):
        scores = cross_val_score(
                     estimator = clf,
                     X = X_test_std,
                     y = y_test,
                     cv = 10,
                     n_jobs = -1,

                     scoring = 'roc_auc'    # AUC
                 )
        print( "AUC <test data> : %0.2f (+/- %0.2f) [%s]" % ( scores.mean(), scores.std(), label) )


    #-------------------------------------------
    # 識別境界
    #-------------------------------------------
    plt.clf()

    for (idx, clf, label) in zip( range( 1,len(all_clf)+2 ),  all_clf, all_clf_labels ):
        print( "識別境界 for ループ idx : ", idx )
        print( "識別境界 for ループ clf : ", clf )

        # idx 番目の plot
        plt.subplot( 2, 3, idx )

        Plot2D.Plot2D.drawDiscriminantRegions( X_combined_std, y_combined, classifier = all_clf[idx-1] )
        plt.title( label )
        plt.xlabel( "Hue [standardized]" )
        plt.ylabel( "Alcohol [standardized]" )
        plt.legend(loc = "best")
        plt.tight_layout()

    plt.savefig("./EnsembleLearning_scikit-learn_8.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()    

    #-------------------------------------------
    # 学習曲線
    #-------------------------------------------
    plt.clf()

    for (idx, clf, label) in zip( range( 1,len(all_clf)+2 ),  all_clf, all_clf_labels ):
        print( "学習曲線 for ループ idx : ", idx )
        print( "学習曲線 for ループ clf : ", clf )

        train_sizes, train_scores, test_scores \
        = learning_curve(
              estimator = clf,    # 推定器 
              X = X_train_std,                              # トレーニングデータでの正解率を計算するため, トレーニングデータを設定
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

        print( "学習曲線 for ループ : \n")
        print( "train_sizes", train_sizes )
        print( "train_means", train_means )
        print( "train_stds", train_stds )
        print( "test_means", test_means )
        print( "test_stds", test_stds )

        # idx 番目の plot
        plt.subplot( 2, 3, idx )
        Plot2D.Plot2D.drawLearningCurve(
            train_sizes = train_sizes,
            train_means = train_means,
            train_stds = train_stds,
            test_means = test_means,
            test_stds = test_stds,
            train_label = "training accuracy",
            test_label = "k-fold cross validation accuracy (cv=10)"
        )
        plt.title( "Learning Curve \n" + label )
        plt.xlabel( "Number of training samples" )
        plt.ylabel( "Accuracy" )
        plt.legend( loc = "best" )
        plt.ylim( [0.8, 1.01] )
        plt.tight_layout()

    plt.savefig("./EnsembleLearning_scikit-learn_9.png", dpi = 300, bbox_inches = 'tight' )
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

    plt.savefig("./EnsembleLearning_scikit-learn_10.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()    
    
    
    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()
