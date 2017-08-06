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
from sklearn.ensemble import RandomForestClassifier     # 

# 自作クラス
import Plot2D


def main():
    print("Enter main()")
    #======================================================================================
    # ランダムフォレスト [random forests] による識別問題（３クラス）
    # アヤメデータの３品種の分類　（Python & scikit-learn ライブラリを使用）   
    # ランダムフォレストによるデータ解析（特徴の重要さ）
    #======================================================================================

    #====================================================
    #   Data Preprocessing（前処理）
    #====================================================
    #----------------------------------------------------
    #   read & set  data
    #----------------------------------------------------
    print("reading data")

    # scikit-learn ライブラリから iris データを読み込む
    iris = datasets.load_iris()
    print(iris.data)

    # 全てのの特徴量を抽出し, dat_X に保管
    dat_X = iris.data

    # クラスラベル（教師データ）を取得
    dat_y = iris.target

    print( 'Class labels:', numpy.unique(dat_y) )    # ※多くの機械学習ライブラリクラスラベルは文字列から整数にして管理されている（最適な性能を発揮するため）
    print("finishing reading data")

    #---------------------------------------------------------------------
    # トレーニングされたモデルの性能評価を未知のデータで評価するために、
    # データセットをトレーニングデータセットとテストデータセットに分割する
    #---------------------------------------------------------------------
    # scikit-learn の cross_validation モジュールの関数 train_test_split() を用いて、70%:テストデータ, 30%:トレーニングデータに分割
    train_test = train_test_split(       # 戻り値:list
                     dat_X, dat_y,       # 
                     test_size = 0.3,    # 0.0~1.0 で指定 
                     random_state = 0    # 
                 )
    
    X_train = train_test[0]
    X_test  = train_test[1]
    y_train = train_test[2]
    y_test  = train_test[3]

    #----------------------------------------------------------------------------------------------------
    # scikit-learn の preprocessing モジュールの StandardScaler クラスを用いて、データをスケーリング
    #----------------------------------------------------------------------------------------------------
    stdScaler = StandardScaler()
    
    # X_train の平均値と標準偏差を計算
    stdScaler.fit( X_train )

    # 求めた平均値と標準偏差を用いて標準化
    X_train_std = stdScaler.transform( X_train )
    X_test_std  = stdScaler.transform( X_test )

    # 分割したデータを行方向に結合（後で plot データ等で使用する）
    X_combined_std = numpy.vstack( (X_train_std, X_test_std) )  # list:(X_train_std, X_test_std) で指定
    y_combined     = numpy.hstack( (y_train, y_test) )

    #====================================================
    #   Learning Process
    #====================================================
    # classifier : random forest \n ( purity = "gini")
    forest = RandomForestClassifier(
                criterion = "gini",     # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 50,      # 弱識別器（決定木）の数
                n_jobs = -1,            # The number of jobs to run in parallel for both fit and predict ( -1 : 全てのCPUコアで並列計算)
                random_state = 1,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    forest.fit( X_train_std, y_train )

    #====================================================
    #   汎化性能の評価
    #====================================================
    # 特徴の重要度の取得
    feature_labels = [ 
                       "sepal length (cm)",
                       "sepal width (cm)",
                       "petal length (cm)",
                       "petal width (cm)"
                     ]

    importances = forest.feature_importances_

    print( importances )
    print( feature_labels )

    for f in range( X_train.shape[1] ):
        print(
            "%2d : %-*s %f" % (f + 1, 30, 
            feature_labels[f], 
            importances[f] )
        )

    #------------------------------------------------------------------------
    # 各特徴の重要度の図示
    #------------------------------------------------------------------------    
    # 現在の図をクリア
    plt.clf()
    #plt.grid(linestyle='-')

    # 各特徴の棒グラフ
    plt.bar(
        range( X_train.shape[1] ),  # x : トレーニング用データを 
        importances,                # y : 
        #color = 'lightblue', 
        align = 'center'
    )

    # x軸の目盛り
    plt.xticks(
        range( X_train.shape[1] ), 
           feature_labels, 
           rotation = 90
    )
    # y軸の目盛り
    plt.axhline( y = 0.1, linewidth = 0.5, color = 'k', linestyle = '--' )
    plt.axhline( y = 0.2, linewidth = 0.5, color = 'k', linestyle = '--' )
    plt.axhline( y = 0.3, linewidth = 0.5, color = 'k', linestyle = '--' )
    plt.axhline( y = 0.4, linewidth = 0.5, color = 'k', linestyle = '--' )
    plt.axhline( y = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )

    plt.xlim( [-1, X_train.shape[1]] )
    plt.title("Feature Importances \n classifier : RandomForest ( n_estimators = 50 )")  # titile
    plt.legend(loc = "upper left")          # 凡例
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig( "./RamdomForest_scikit-learn_4.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()
    

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
