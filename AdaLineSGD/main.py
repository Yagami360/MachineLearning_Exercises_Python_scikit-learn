# -*- coding: utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt
import pandas

# 自作クラス
from MLPlot import MLPlot          # 機械学習用の図の描写をサポートする関数群からなるクラス
import AdaLineSGD

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
    df_Iris.tail()

    y_labels = df_Iris.iloc[0:100,4].values                       #
    y_labels = numpy.where( y_labels == "Iris-setosa", -1, 1 )    # Iris-setosa = -1, Iris-virginica = 1 にラベル変換
    X_features = df_Iris.iloc[0:100, [0,2]].values                # pandas DataFrame のrow,columnの指定方法 iloc:rawのindex(0 ~ ),
                                                                  # columnのindex(0 ~ )）
    #----------------------------------------------------
    #   Pre Process（前処理）
    #----------------------------------------------------
    # 学習データを正規化
    X_features_std = numpy.copy( X_features )                                                   # ディープコピー（参照コピーではない）
    X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std()  # 0列目全てにアクセス[:,0]
    X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()

    #----------------------------------------------------
    #   Learning Process
    #----------------------------------------------------
    ada1 = AdaLineSGD.AdaLineSGD( lRate = 0.01, numIter = 30, random_state = 1 )
    ada1.fit( X_train = X_features_std, y_train = y_labels )
    
    #----------------------------------------------------
    #   Draw Figure 1
    #----------------------------------------------------
    # plot learning data
    plt.subplot(2,2,1) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')

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

    # plot fitting
    plt.subplot(2,2,2) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')
    
    plt.plot(
        range(1, len(ada1.cost_) + 1), ada1.cost_, 
        marker = 'o'
    )

    plt.xlabel("Epochs")
    plt.ylabel("Avarage cost")
    plt.title( "AdalineSGD (Learning rate 0.01)" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # plot result
    plt.subplot(2,2,3) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.title("AdalineSGD (Learning rate 0.01)" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    MLPlot.drawDiscriminantRegions( X_features_std, y_labels, classifier = ada1 )

    # save & show
    plt.savefig("./AdalineSGD_1.png", dpi=300)
    plt.show()

    #----------------------------------------------------
    #   Lerning Process init
    #----------------------------------------------------
    ada2 = AdaLineSGD.AdaLineSGD( lRate = 0.01, numIter = 5, random_state = 1 )
    ada2.fit( X_train = X_features_std, y_train = y_labels )
        
    #----------------------------------------------------
    #   Draw Figure 2-1
    #----------------------------------------------------
    # plot fitting
    plt.subplot(2,2,1) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')
    plt.plot(
        range(1, len(ada2.cost_) + 1), 
        ada2.cost_, 
        marker = 'o'
    )
    plt.xlabel("Epochs")
    plt.ylabel("Avarage cost")
    plt.title( "AdalineSGD before online learning" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # plot result
    plt.subplot(2,2,2) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.title("AdalineSGD before online learning" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    MLPlot.drawDiscriminantRegions( X_features_std, y_labels, classifier = ada2 )

    #----------------------------------------------------
    #   Online Learning Process
    #----------------------------------------------------
    # （擬似的な）ストリーミングデータ (5~10) で、（擬似的な）オンライン学習
    for smIndex in range(5,100):
        print(smIndex)
        ada2.online_fit( X_train = X_features_std[0:smIndex, :], y_train = y_labels[0:smIndex] )
    
    #----------------------------------------------------
    #   Draw Figure 2-2
    #----------------------------------------------------
    plt.subplot(2,2,3) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')
    plt.plot(
        range(1, len(ada2.cost_) + 1), ada2.cost_, 
        marker = 'o',
        markersize = 2
    )
    plt.xlabel("Epochs")
    plt.ylabel("Avarage cost (add online learning)")
    plt.title( "AdalineSGD after online learning" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # plot result
    plt.subplot(2,2,4) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.xlabel("sepal length [Normalized]")
    plt.ylabel("petal length [Normalized]")        # label
    plt.title("AdalineSGD after online learning" )
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    #ada2.weights_[0] = ada1.weights_[0]
    #ada2.weights_[1] = ada1.weights_[1]
    #ada2.weights_[2] = ada1.weights_[2]

    MLPlot.drawDiscriminantRegions( X_features_std, y_labels, classifier = ada2 )


    # save & show
    plt.savefig("./AdalineSGD_2.png", dpi=300)
    plt.show()

if __name__ == '__main__':
     main()
