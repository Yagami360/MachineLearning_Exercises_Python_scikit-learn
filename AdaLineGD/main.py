# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import pandas
import Plot2D
import AdaLineGD

def main():
    #----------------------------------------------------
    #   read & set iris data
    #----------------------------------------------------
    # pandasライブラリを使用して Iris データを読み込み (dataframe obj)
    print("reading iris data in pandas")
    df_Iris = pandas.read_csv(
        'https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', 
        header = None
    )
    print("finish reading iris data in pandas")
    #df_Iris.tail()

    dat_y = df_Iris.iloc[ 0:100,4 ].values              #
    dat_y = numpy.where( dat_y=="Iris-setosa", -1, 1 )  # Iris-setosa = -1, Iris-virginica = 1 に変換
    dat_X = df_Iris.iloc[ 0:100, [0,2] ].values         # pandas DataFrame のrow,columnの指定方法（iloc:rawのindex(0 ~ ), columnのindex(0 ~ )）

    #----------------------------------------------------
    #   Draw learing data
    #----------------------------------------------------
    plt.subplot( 2,2,1 ) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid( linestyle='-' )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 品種 setosa のplot(赤の○)
    plt.scatter(
        dat_X[0:50,0], dat_X[0:50,1],
        color = "red",
        marker = "o",
        label = "setosa"
    )
    # 品種 virginica のplot(青のx)
    plt.scatter(
        dat_X[50:100,0], dat_X[50:100,1],
        color = "blue",
        marker = "x",
        label = "virginica"
    )

    plt.title( "Learning data" )        #
    plt.xlabel( "sepal length [cm]" )
    plt.ylabel( "petal length [cm]" )   # label
    plt.legend( loc = "upper left" )    # 凡例

    #----------------------------------------------------
    #   set AdaLineGD & draw
    #----------------------------------------------------
    ada1 = AdaLineGD.AdaLineGD( lRate = 0.01, numIter=10 )
    ada1.fit( X_train = dat_X, y_train = dat_y )
    
    ada2 = AdaLineGD.AdaLineGD( lRate = 0.001, numIter=10 )
    ada2.fit( X_train = dat_X, y_train = dat_y )

    plt.subplot( 2,2,2 ) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid( linestyle='-' )
    plt.plot(
        range( 1, len(ada1.cost_) + 1 ), 
        numpy.log10(ada1.cost_), 
        marker = 'o'
    )
    plt.xlabel("Epochs")
    plt.ylabel("log(Sum-squared-error)")
    plt.title("Adaline - Learning rate 0.01")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.subplot( 2,2,3 ) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid( linestyle='-' )
    plt.plot(
        range( 1, len(ada2.cost_) + 1 ), 
        numpy.log10(ada2.cost_), 
        marker = 'o'
    )
    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")
    plt.title("Adaline - Learning rate 0.001")
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    #----------------------------------------------

    #-----------------------------------------------
    plt.savefig( "./AdalineGD_1.png", dpi=300 )
    plt.show()

if __name__ == '__main__':
     main()