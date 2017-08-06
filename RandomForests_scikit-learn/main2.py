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
    # ランダムフォレストによるデータ解析（各特徴の誤り率、OBB error rate, 特徴の重要さ等）
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

    # 3,4 列目の特徴量を抽出し、dat_X に保管
    dat_X = iris.data[ :, [2,3] ]

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

    # 学習データを正規化（後で plot データ等で使用する）
    dat_X_std = numpy.copy(dat_X)                                           # ディープコピー（参照コピーではない）
    dat_X_std[:,0] = ( dat_X[:,0] - dat_X[:,0].mean() ) / dat_X[:,0].std()  # 0列目全てにアクセス[:,0]
    dat_X_std[:,1] = ( dat_X[:,1] - dat_X[:,1].mean() ) / dat_X[:,1].std()

    #====================================================
    #   Learning Process & 汎化性能の評価
    #====================================================
    # classifier : random forest \n ( purity = "gini")
    forest = RandomForestClassifier(
                criterion = "gini",     # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                #n_estimators = 2,       # 弱識別器（決定木）の数
                n_jobs = -1,            # The number of jobs to run in parallel for both fit and predict (全てのCPUコアで並列計算)
                random_state = 1,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    #forest.fit( X_train_std, y_train )

    max_numForests = 50
    lst_numForests = range( 1, max_numForests )     # 森のサイズのリスト（1 ~ max_numForests）
    
    obb_errors = []            # OOB error rate のリスト（森のサイズに対するリスト）
    iris_errors = []           # テストに使用した Iris data 全体での誤り率のリスト （森のサイズに対するリスト）
    setosa_errors = []         # 品種 setosa の誤り率のリスト（森のサイズに対するリスト）
    virginica_errors = []      # 品種 virginica の誤り率のリスト（森のサイズに対するリスト）
    versicolor_errors = []     # 品種 versicolor の誤り率のリスト（森のサイズに対するリスト）

    for i in lst_numForests:
        # 森のサイズ（弱識別器数[決定木の数]）を設定後、学習データで学習させる。
        forest.set_params( n_estimators = i )
        forest.fit( X_train_std, y_train )

        # （この森のサイズのランダムフォレストでの）OOB誤り率の計算
        obb_error = 1 - forest.oob_score_
        obb_errors.append( obb_error )
        
        # （この森のサイズのランダムフォレストでの）各特徴量の計算
        y_predict            = forest.predict( X_test_std )
        #y_predict_setosa     = forest.predict( dat_X_std[0:50] )
        #y_predict_virginica  = forest.predict( dat_X_std[51:100] )
        #y_predict_versicolor = forest.predict( dat_X_std[101:150] )

        error_rate           = 1 - accuracy_score( y_test, y_predict )
        #setosa_error         = 1 - accuracy_score( dat_y, y_predict_setosa )
        #virginica_error      = 1 - accuracy_score( dat_y, y_predict_virginica )
        #versicolor_error     = 1 - accuracy_score( dat_y, y_predict_versicolor )

        iris_errors.append( error_rate )
        #setosa_errors.append( setosa_error )
        #virginica_errors.append( virginica_error )
        #versicolor_errors.append( versicolor_error )
        
        print( "[error rate]")
        print( "the number of forests = ", i )
        print( "obb_error = ", obb_error )
        print( "error_rate = ", error_rate )

    #------------------------------------------------------------------------
    # 各特徴の誤り率＆OOB 誤り率 [ out-of-bag error rate] の図示
    #------------------------------------------------------------------------    
    # 現在の図をクリア
    plt.clf()
    plt.grid()

    # OBB error rate plot(黒の●)
    plt.plot(
        lst_numForests, obb_errors,
        color = "black",
        marker = "o",
        label = "OOB error rate",
        markersize = 6,                 # プロット点サイズ
        linestyle = "solid"             # 実線
    )

    # iris error rate plot
    plt.plot(
        lst_numForests, iris_errors,
        color = "cyan",
        marker = "v",
        label = "iris error rate",
        markersize = 6,                 # プロット点サイズ
        linestyle = "solid"             # 実線
    )
    
    """
    # 品種 setosa の誤り率の plot(赤の□)
    plt.plot(
        lst_numForests, setosa_errors,
        color = "red",
        marker = "s",
        label = "setosa error rate",
        markersize = 6,                 # プロット点サイズ
        linestyle = "solid"             # 実線
    )
    # 品種 virginica の誤り率の plot(青のx)
    plt.plot(
        lst_numForests, virginica_errors,
        color = "blue",
        marker = "x",
        label = "virginica error rate",
        markersize = 6,                 # プロット点サイズ
        linestyle = "solid"             # 実線
    )
    # 品種 versicolor の誤り率の plot(緑の+)
    plt.plot(
        lst_numForests, versicolor_errors,
        color = "green",
        marker = "+",
        label = "versicolor error rate",
        markersize = 6,                 # プロット点サイズ
        linestyle = "solid"             # 実線
    )
    """

    plt.title("Error rate of Iris data (classifier : RandomForest)")  # titile
    plt.xlabel("The number of trees in the forest") # label x-axis
    plt.ylabel("Error rate") # label y-axis
    plt.legend(loc = "upper left")          # 凡例
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig("./RamdomForest_scikit-learn_3.png", dpi=300)
    plt.show()
    

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
