# -*- coding: utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt
import pandas

# 自作クラス
from MLPlot import MLPlot       # 機械学習用の図の描写をサポートする関数群からなるクラス
import AdaLineGD

def main():
    #----------------------------------------------------
    #   read & set iris data
    #----------------------------------------------------
    # pandasライブラリを使用して Iris データを読み込み (dataframe obj)
    print("reading iris data from pandas-lib")
    df_Iris = pandas.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', 
        header = None)
    print("finish reading iris data from pandas-lib")
    #df_Iris.tail()

    y_labels = df_Iris.iloc[0:100,4].values                     #
    y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1 )  # Iris-setosa = -1, Iris-virginica = 1 に変換
    X_features = df_Iris.iloc[0:100, [0,2]].values              # pandas DataFrame のrow,columnの指定方法（iloc:rawのindex(0 ~ ),
                                                                # columnのindex(0 ~ )）

    #----------------------------------------------------
    #   Draw learning data 1 (UnNormalized)
    #----------------------------------------------------
    plt.subplot(2,2,1) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')

    # 品種 setosa のplot(赤の○)
    plt.scatter(
        X_features[0:50,0], X_features[0:50,1],
        color = "red",
        marker = "o",
        label = "setosa"
    )

    # 品種 virginica のplot(青のx)
    plt.scatter(
        X_features[50:100,0], X_features[50:100,1],
        color = "blue",
        marker = "x",
        label = "virginica"
    )

    plt.title("Learning data [UnNormalized]")        #
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")   # label
    plt.legend(loc = "upper left")    # 凡例
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    #----------------------------------------------------
    #   set AdaLineGD & draw 1
    #----------------------------------------------------
    ada1 = AdaLineGD.AdaLineGD( lRate = 0.01, numIter = 30 )
    ada1.fit( X_train = X_features, y_train = y_labels )
    
    ada2 = AdaLineGD.AdaLineGD( lRate = 0.0001, numIter = 30 )
    ada2.fit( X_train = X_features, y_train = y_labels )

    plt.subplot(2,2,2) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')
    
    plt.plot(
        range(1, len(ada1.cost_) + 1), numpy.log10(ada1.cost_), 
        marker = 'o'
    )

    plt.xlabel("Epochs")
    plt.ylabel("log(Sum-squared-error)")
    plt.title("Adaline - Learning rate 0.01")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.subplot(2,2,3) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')

    plt.plot(
        range(1, len(ada2.cost_) + 1), ada2.cost_, 
        marker = 'o'
    )

    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")
    plt.title("Adaline - Learning rate 0.0001")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.subplot(2,2,4) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.title("Adaline - Learning rate 0.0001")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    MLPlot.drawDiscriminantRegions( X_features, y_labels, ada2 )
    
    plt.savefig("./AdalineGD_1.png", dpi=300)
    plt.show()


    #----------------------------------------------------
    #   normalized lerning data & draw
    #----------------------------------------------------
    X_features_std = numpy.copy( X_features )                                                          # ディープコピー（参照コピーではない）
    X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std()  # 0列目全てにアクセス[:,0]
    X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()

    plt.subplot(2,2,1) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid( linestyle='-' )

    # 品種 setosa のplot(赤の○)
    plt.scatter(
        X_features_std[0:50,0], X_features_std[0:50,1],
        color = "red",
        marker = "o",
        label = "setosa"
    )

    # 品種 virginica のplot(青のx)
    plt.scatter(
        X_features_std[50:100,0], X_features_std[50:100,1],
        color = "blue",
        marker = "x",
        label = "virginica"
    )

    plt.title("Learning data [Normalized]")        #
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.legend(loc = "upper left")    # 凡例    
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    #----------------------------------------------------
    #   set AdaLineGD & draw 2 (Normalized)
    #----------------------------------------------------
    ada3 = AdaLineGD.AdaLineGD( lRate = 0.01, numIter = 30 )
    ada3.fit( X_features_std, y_labels )
   
    plt.subplot(2,2,2) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')

    plt.plot(
        range(1, len(ada3.cost_) + 1), ada3.cost_, 
        marker = 'o'
    )

    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")
    plt.title("Adaline - Learning rate 0.01")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.subplot(2,2,3) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.title("Adaline - Learning rate 0.01")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    MLPlot.drawDiscriminantRegions( X_features_std, y_labels, ada3 )

    plt.savefig("./AdalineGD_2.png", dpi=300)
    plt.show()

    #---------------------------------------------------
    #
    #---------------------------------------------------

if __name__ == '__main__':
     main()
