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
from MLPlot import MLPlot                               # 機械学習用の図の描写をサポートする関数群からなるクラス

def main():
    print("Enter main()")
    #==================================================================================
    #   RBF Kernel C-SVM を用いた非線形分離問題（アヤメデータの３クラス）
    #==================================================================================
    
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
    X_features = iris.data[ :, [2,3] ]

    # クラスラベル（教師データ）を取得
    y_labels = iris.target

    print( 'Class labels:', numpy.unique(y_labels) )    # ※多くの機械学習ライブラリクラスラベルは文字列から整数にして管理されている（最適な性能を発揮するため）
    print("finishing reading data")

    #---------------------------------------------------------------------
    # トレーニングされたモデルの性能評価を未知のデータで評価するために、
    # データセットをトレーニングデータセットとテストデータセットに分割する
    #---------------------------------------------------------------------
    # scikit-learn の cross_validation モジュールの関数 train_test_split() を用いて、70%:テストデータ, 30%:トレーニングデータに分割
    train_test = train_test_split(               # 戻り値:list
                     X_features, y_labels,       # 
                     test_size = 0.3,            # 0.0~1.0 で指定 
                     random_state = 0            # 
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
    X_features_std = numpy.copy( X_features )                                           # ディープコピー（参照コピーではない）
    X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std()  # 0列目全てにアクセス[:,0]
    X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()

    #====================================================
    #   Learning Process
    #====================================================
    kernelSVM1 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 0.20,       # RFBカーネル関数のγ値
        C = 1,              # C-SVM の C 値
        probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )
    kernelSVM1.fit( X_train_std, y_train )

    kernelSVM2 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 0.20,       # RFBカーネル関数のγ値
        C = 10,             # C-SVM の C 値
        probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )
    kernelSVM2.fit( X_train_std, y_train )

    kernelSVM3 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 100,        # RFBカーネル関数のγ値
        C = 1,              # C-SVM の C 値
        probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )
    kernelSVM3.fit( X_train_std, y_train )
        
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
        X_features_std[0:50,0], X_features_std[0:50,1],
        color = "red",
        edgecolor = 'black',
        marker = "s",
        label = "setosa"
    )
    # 品種 virginica のplot(青のx)
    plt.scatter(
        X_features_std[51:100,0], X_features_std[51:100,1],
        color = "blue",
        edgecolor = 'black',
        marker = "x",
        label = "virginica"
    )
    # 品種 versicolor のplot(緑の+)
    plt.scatter(
        X_features_std[101:150,0], X_features_std[101:150,1],
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
    # classifier1 : kernelSVM1 [RBF-Kernel C-SVM (C=1.0,σ=0.2)]
    plt.subplot(2,2,2)
    MLPlot.drawDiscriminantRegions( 
        X_features = X_combined_std, y_labels = y_combined,
        classifier = kernelSVM1,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result(C=1,σ=0.2)")         # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    # classifier2 : kernelSVM2 [RBF-Kernel C-SVM (C=10.0,σ=0.2)]
    plt.subplot(2,2,3)
    MLPlot.drawDiscriminantRegions( 
        X_features = X_combined_std, y_labels = y_combined,
        classifier = kernelSVM2,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result(C=10,σ=0.2)")         # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # classifier3 : kernelSVM3 [RBF-Kernel C-SVM (C=1.0,σ=100)]
    plt.subplot(2,2,4)
    MLPlot.drawDiscriminantRegions( 
        X_features = X_combined_std, y_labels = y_combined,
        classifier = kernelSVM3,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result(C=1,σ=100)")         # titile
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 
    plt.savefig("./SVM_scikit-learn_5.png", dpi=300)
    plt.show()

    #-------------------------------
    # 識別率を計算＆出力
    #-------------------------------
    y_predict1 = kernelSVM1.predict( X_test_std )
    y_predict2 = kernelSVM2.predict( X_test_std )
    y_predict3 = kernelSVM3.predict( X_test_std )

    print("<テストデータの識別結果>")
    
    print("classifier1 : kernelSVM1 [RBF-Kernel C-SVM (C=1.0,σ=0.2)]")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict1).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict1) )

    print("classifier2 : kernelSVM2 [RBF-Kernel C-SVM (C=10.0,σ=0.2)]")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict2).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict2) )

    print("classifier3 : kernelSVM3 [RBF-Kernel C-SVM (C=1.0,σ=100)]")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict3).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え
    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict3) )

    #--------------------------------------------------------------------------------------------------------
    # predict_proba() 関数を使用して、指定したサンプルのクラスの所属関係を予想
    # 戻り値は、サンプルが Iris-Setosa, Iris-Versicolor, Iris-Virginica に所属する確率をこの順で表している.
    #--------------------------------------------------------------------------------------------------------
    preProb = []

    # classifier1 : kernelSVM1 [RBF-Kernel C-SVM (C=1.0,σ=0.2)]
    preProb.append( kernelSVM1.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM1.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM1.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納

    # classifier2 : kernelSVM2 [RBF-Kernel C-SVM (C=10.0,σ=0.2)]
    preProb.append( kernelSVM2.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM2.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM2.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納

    # classifier3 : kernelSVM2 [RBF-Kernel C-SVM (C=1.0,σ=100)]
    preProb.append( kernelSVM3.predict_proba( X_test_std[0, :].reshape(1, -1) ) )  # 0 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM3.predict_proba( X_test_std[1, :].reshape(1, -1) ) )  # 1 番目のテストデータを reshap でタプル化して格納
    preProb.append( kernelSVM3.predict_proba( X_test_std[2, :].reshape(1, -1) ) )  # 2 番目のテストデータを reshap でタプル化して格納
    
    # 各々のサンプルの所属クラス確率の出力
    print("classifier1 : kernelSVM1 [RBF-Kernel C-SVM (C=1.0,σ=0.2)]")
    print("サンプル 0 の所属クラス確率 [%] :", preProb[0] * 100 )
    print("サンプル 1 の所属クラス確率 [%] :", preProb[1] * 100 )
    print("サンプル 2 の所属クラス確率 [%] :", preProb[2] * 100 )

    print("classifier2 : kernelSVM2 [RBF-Kernel C-SVM (C=10.0,σ=0.2)]")
    print("サンプル 0 の所属クラス確率 [%] :", preProb[3] * 100 )
    print("サンプル 1 の所属クラス確率 [%] :", preProb[4] * 100 )
    print("サンプル 2 の所属クラス確率 [%] :", preProb[5] * 100 )

    print("classifier3 : kernelSVM3 [RBF-Kernel C-SVM (C=1.0,σ=100)]")
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
    plt.savefig("./SVM_scikit-learn_6.png", dpi=300)
    plt.show()
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
