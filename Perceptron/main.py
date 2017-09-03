# -*- coding: utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt
import pandas

# 自作クラス
import Perceptron
from MLPlot import MLPlot


def main():
    #----------------------------------------------------
    #   read & set iris data
    #----------------------------------------------------
    print("reading iris data from pandas-lib")
    # pandasライブラリを使用して Iris データを読み込み (dataframe obj)
    df_Iris = pandas.read_csv(
        'https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', 
        header = None
    )
    print("finish reading iris data from pandas-lib")
    #df_Iris.tail()

    y_labels = df_Iris.iloc[ 0:100,4 ].values                   #
    y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1 )  # Iris-setosa = -1, Iris-virginica = 1 に変換
    X_features = df_Iris.iloc[ 0:100, [0,2] ].values            # pandas DataFrame のrow,columnの指定方法（iloc:rawのindex(0 ~ ), columnのindex(0 ~ )）

    #----------------------------------------------------
    #   Draw learing data
    #----------------------------------------------------
    plt.subplot( 2,2,1 ) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

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

    plt.title( "Learning data" )        #
    plt.xlabel( "sepal length [cm]" )
    plt.ylabel( "petal length [cm]" )   # label
    plt.legend( loc = "upper left" )    # 凡例

    #----------------------------------------------------
    #   set perceptron & draw
    #----------------------------------------------------
    ppn1 = Perceptron.Perceptron( lRate=0.1, numIter=10 )
    ppn1.fit( X_features, y_labels )

    plt.subplot( 2,2,2 )    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.tight_layout()      # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    plt.plot(
        range( 1,len(ppn1.errors_) + 1 ), ppn1.errors_, 
        marker = 'o'
    )
    plt.title( "Fitting" )
    plt.xlabel("Epochs (Number of trainings)")
    plt.ylabel("Number of updates")
    
    #----------------------------------------------
    plt.subplot( 2,2,3 )    # plt.subplot(行数, 列数, 何番目のプロットか)

    MLPlot.drawDiscriminantRegions( X_features, y_labels, classifier=ppn1 )
    plt.title( "Result of discrimination" )
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend( loc="upper left" )
    plt.tight_layout()      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    #-----------------------------------------------
    plt.savefig( "./Perceptron_1.png", dpi=300 )
    plt.show()

if __name__ == '__main__':
    main()
