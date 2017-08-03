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
    #==================================================================================
    # ランダムフォレスト [random forests] による識別問題（３クラス）
    # アヤメデータの３品種の分類　（Python & scikit-learn ライブラリを使用）   
    #==================================================================================

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
    #   Learning Process
    #====================================================
    # classifier1 : random forest1 \n ( purity = "entropy", The number of trees in the forest = 2 )
    forest1 = RandomForestClassifier(
                criterion = "entropy",  # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 2,       # 弱識別器（決定木）の数
                n_jobs = 2,             # The number of jobs to run in parallel for both fit and predict
                random_state = 1,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    forest1.fit( X_train_std, y_train )

    # classifier2 : random forest2 \n ( purity = "entropy", The number of trees in the forest = 5 )
    forest2 = RandomForestClassifier(
                criterion = "entropy",  # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 5,       # 弱識別器（決定木）の数
                n_jobs = 2,             # The number of jobs to run in parallel for both fit and predict
                random_state = 1,       # 
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    forest2.fit( X_train_std, y_train )
     
    # classifier3 : random forest3 \n ( purity = "entropy", The number of trees in the forest = 10 )
    forest3 = RandomForestClassifier(
                criterion = "entropy",  # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 10,      # 弱識別器（決定木）の数
                n_jobs = 2,             # The number of jobs to run in parallel for both fit and predict
                random_state = 1,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )
    forest3.fit( X_train_std, y_train )
    

    #====================================================
    #   汎化性能の評価
    #====================================================
    #-------------------------------
    # サンプルデータの plot
    #-------------------------------
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2,2,1)
    plt.grid(linestyle='-')

    # 品種 setosa のplot(赤の□)
    plt.scatter(
        dat_X_std[0:50,0], dat_X_std[0:50,1],
        color = "red",
        edgecolor = 'black',
        marker = "s",
        label = "setosa"
    )
    # 品種 virginica のplot(青のx)
    plt.scatter(
        dat_X_std[51:100,0], dat_X_std[51:100,1],
        color = "blue",
        edgecolor = 'black',
        marker = "x",
        label = "virginica"
    )
    # 品種 versicolor のplot(緑の+)
    plt.scatter(
        dat_X_std[101:150,0], dat_X_std[101:150,1],
        color = "green",
        edgecolor = 'black',
        marker = "+",
        label = "versicolor"
    )

    plt.title("iris data [Normalized]")     # title
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    #-------------------------------
    # 識別結果＆識別領域の表示
    #-------------------------------
    # classifier1 : 
    plt.subplot(2,2,2)
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_combined_std, dat_y = y_combined,
        classifier = forest1,
        list_test_idx = range( 101,150 )
    )
    plt.title("Ramdom Forest 1 \n ( purity = entropy, The number of trees in the forest = 2 )")   # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    # classifier2 : 
    plt.subplot(2,2,3)
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_combined_std, dat_y = y_combined,
        classifier = forest2,
        list_test_idx = range( 101,150 )
    )
    plt.title("Ramdom Forest 2 \n ( purity = entropy, The number of trees in the forest = 5 )")   # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # classifier3 : 
    plt.subplot(2,2,4)
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_combined_std, dat_y = y_combined,
        classifier = forest3,
        list_test_idx = range( 101,150 )
    )
    plt.title("Ramdom Forest 3 \n ( purity = entropy, The number of trees in the forest = 10 )")  # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 
    plt.savefig("./RandomForest_scikit-learn_1.png", dpi=300)
    plt.show()

    #-------------------------------
    # 識別率を計算＆出力
    #-------------------------------
    y_predict1 = forest1.predict( X_test_std )
    y_predict2 = forest2.predict( X_test_std )
    y_predict3 = forest3.predict( X_test_std )

    print("<テストデータの識別結果>")
    
    print("classifier1 : Ramdom Forest1 \n ( purity = entropy, The number of trees in the forest = 2 )")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict1).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict1) )

    print("classifier2 : Ramdom Forest2 \n ( purity = entropy, The number of trees in the forest = 5 )")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict2).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict2) )

    print("classifier3 : Ramdom Forest3 \n ( purity = entropy, The number of trees in the forest = 10 )")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict3).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict3) )

    #--------------------------------------------------------------------------------------------------------
    # predict_proba() 関数を使用して、指定したサンプルのクラスの所属関係を予想
    # 戻り値は、サンプルが Iris-Setosa, Iris-Versicolor, Iris-Virginica に所属する確率をこの順で表している.
    #--------------------------------------------------------------------------------------------------------
    preProb = []

    # classifier1 : 
    preProb.append( forest1.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest1.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest1.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納

    # classifier2 : 
    preProb.append( forest2.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest2.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest2.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納

    # classifier3 : 
    preProb.append( forest3.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest3.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( forest3.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納
    
    # 各々のサンプルの所属クラス確率の出力
    print("classifier1 : Ramdom Forest1 ( purity = entropy, The number of trees in the forest = 2 )")
    print("サンプル 0 の所属クラス確率 [%] :", preProb[0] * 100 )
    print("サンプル 1 の所属クラス確率 [%] :", preProb[1] * 100 )
    print("サンプル 2 の所属クラス確率 [%] :", preProb[2] * 100 )

    print("classifier2 : Ramdom Forest2 ( purity = entropy, The number of trees in the forest = 5 )")
    print("サンプル 0 の所属クラス確率 [%] :", preProb[3] * 100 )
    print("サンプル 1 の所属クラス確率 [%] :", preProb[4] * 100 )
    print("サンプル 2 の所属クラス確率 [%] :", preProb[5] * 100 )

    print("classifier3 : Ramdom Forest3 ( purity = entropy, The number of trees in the forest = 10 )")
    print("サンプル 0 の所属クラス確率 [%] :", preProb[6] * 100 )
    print("サンプル 1 の所属クラス確率 [%] :", preProb[7] * 100 )
    print("サンプル 2 の所属クラス確率 [%] :", preProb[8] * 100 )
    
    #------------------------------------------------------------------------
    # 各々のサンプルの所属クラス確率の図示
    #------------------------------------------------------------------------
    # 現在の図をクリア
    plt.clf()

    # 所属クラスの確率を棒グラフ表示
    k = 0
    for i in range(3):
        for j in range(3):
            k += 1
            print("棒グラフ生成（複数図）", i,j,k)
            plt.subplot( 3, 3, k )                                    # plt.subplot(行数, 列数, 何番目のプロットか)
            plt.title("samples[ %d ]" % j + " by classifier %d " % (i+1) )          # title
            plt.xlabel("Varieties (Belonging class)")                 # label x-axis
            plt.ylabel("probability[%]")                              # label y-axis
            plt.ylim( 0, 100 )                                        # y軸の範囲(0~100)
            plt.legend(loc = "upper left")                            # 凡例    

            # 棒グラフ
            plt.bar(
                left = [0,1,2],
                height = preProb[k-1][0] * 100,
                tick_label = ["Setosa","Versicolor","Virginica"]
            )             
            plt.tight_layout()                                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    
    # 図の保存＆表示
    plt.savefig("./RamdomForest_scikit-learn_2.png", dpi=300)
    plt.show()
    
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
