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
from sklearn.svm import SVC                             # 

# 自作クラス
import Plot2D

def main():

    #==================================================================================
    #   Liner SVM
    #==================================================================================
    print("Start Process1")
    
    #====================================================
    #   Pre Process（前処理）
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
    """
    # train_test_split() の戻り値の確認
    print("train_test[0]:", train_test[0])  # X_tarin
    print("train_test[1]:", train_test[1])  # X_test
    print("train_test[2]:", train_test[2])  # y_train
    print("train_test[3]:", train_test[3])  # y_test
    print("train_test[4]:", train_test[4])  # inavlid value
    print("train_test[5]:", train_test[5])  # inavlid value
    """
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
    # 線形SVMのオブジェクトを生成する。
    linearSVM = SVC( 
                    kernel = "linear",      # kernel = "linearSVM" : 線形SVM
                    C = 1.0,                # C-SVM の C 値
                    random_state=0,         #
                    probability = True      # 学習後の predict_proba method による予想確率を有効にする
                )    
    linearSVM.fit( X_train_std, y_train )
    
    #====================================================
    #   汎化性能の評価
    #====================================================
    #-------------------------------
    # サンプルデータの plot
    #-------------------------------
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(1,2,1)
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
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(1,2,2)
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_combined_std, dat_y = y_combined,
        classifier = linearSVM,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result")         # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.savefig("./SVM_scikit-learn_1.png", dpi=300)
    #plt.show()

    #-------------------------------
    # 識別率を計算＆出力
    #-------------------------------
    y_predict = linearSVM.predict( X_test_std )

    # 誤分類のサンプル数を出力
    print( 'Misclassified samples: %d' % (y_test != y_predict).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え

    # 分類の正解率を出力
    print( 'Accuracy: %.2f' % accuracy_score(y_test, y_predict) )

    #--------------------------------------------------------------------------------------------------------
    # predict_proba() 関数を使用して、指定したサンプルのクラスの所属関係を予想
    # 戻り値は、サンプルが Iris-Setosa, Iris-Versicolor, Iris-Virginica に所属する確率をこの順で表している.
    #--------------------------------------------------------------------------------------------------------
    pre0 = linearSVM.predict_proba( X_test_std[0, :].reshape(1, -1) )   # 0番目のテストデータをreshap でタプル化して渡す
    pre1 = linearSVM.predict_proba( X_test_std[1, :].reshape(1, -1) )   # 1番目のテストデータをreshap でタプル化して渡す
    pre2 = linearSVM.predict_proba( X_test_std[2, :].reshape(1, -1) )   # 2番目のテストデータをreshap でタプル化して渡す
    pre3 = linearSVM.predict_proba( X_test_std[3, :].reshape(1, -1) )   # 3番目のテストデータをreshap でタプル化して渡す
    
    print("サンプル0の所属クラス確率 :", pre0[0]*100 )
    print("サンプル1の所属クラス確率 :", pre1[0]*100 )
    print("サンプル2の所属クラス確率 :", pre2[0]*100 )
    print("サンプル3の所属クラス確率 :", pre3[0]*100 )

    #------------------------------------------------------------------------
    # 各々のサンプルの所属クラスの図示 ["Setosa","Versicolor","Virginica"]
    #------------------------------------------------------------------------
    # 現在の図をクリア
    plt.clf()

    # 所属クラスの確率を棒グラフ表示(1,1)
    plt.subplot(2,2,1)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.title("Probability of class (use predict_proba method)")
    plt.xlabel("Varieties (Belonging class)")   # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.legend(loc = "upper left")              # 凡例    

    # 棒グラフ
    plt.bar(
        left = [0,1,2],
        height  = pre0[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    # 所属クラスの確率を棒グラフ表示(1,2)
    plt.subplot(2,2,2)
    plt.xlabel("Varieties (Belonging class)")   # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre1[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,3)
    plt.xlabel("Varieties (Belonging class)")   # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre2[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,4)
    plt.xlabel("Varieties (Belonging class)")   # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre3[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig("./SVM_scikit-learn_2.png", dpi=300)
    plt.show()
    
    #==================================================================================================
    # RBF-Kernelを使用したSVMによる非線形問題
    #==================================================================================================
    print("Start Process2")
    #====================================================
    #   Pre Process（前処理）
    #====================================================
    #----------------------------------------------------
    #   read & set  data (randam data)
    #----------------------------------------------------
    # 乱数の seed
    numpy.random.seed(0)



    #    
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

