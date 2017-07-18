# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import pandas
import Plot2D
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

    dat_y = df_Iris.iloc[0:100,4].values                #
    dat_y = numpy.where(dat_y == "Iris-setosa", -1, 1)  # Iris-setosa = -1, Iris-virginica = 1 にラベル変換
    dat_X = df_Iris.iloc[0:100, [0,2]].values           # pandas DataFrame のrow,columnの指定方法 iloc:rawのindex(0 ~ ),
                                                        # columnのindex(0 ~ )）
    #----------------------------------------------------
    #   Pre Process（前処理）
    #----------------------------------------------------
    # 学習データを正規化
    dat_X_std = numpy.copy(dat_X)                                           # ディープコピー（参照コピーではない）
    dat_X_std[:,0] = ( dat_X[:,0] - dat_X[:,0].mean() ) / dat_X[:,0].std()  # 0列目全てにアクセス[:,0]
    dat_X_std[:,1] = ( dat_X[:,1] - dat_X[:,1].mean() ) / dat_X[:,1].std()

    #----------------------------------------------------
    #   Learning Process
    #----------------------------------------------------
    ada1 = AdaLineSGD.AdaLineSGD( lRate = 0.01, numIter = 30, random_state = 1 )
    ada1.fit( X_train = dat_X_std, y_train = dat_y )
    
    #----------------------------------------------------
    #   Draw Figure 1
    #----------------------------------------------------
    # plot learning data
    plt.subplot(2,2,1) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')

    # 品種 setosa のplot(赤の○)
    plt.scatter(
        dat_X_std[0:50,0], dat_X_std[0:50,1],
        color = "red",
        marker = "o",
        label = "setosa"
    )
    # 品種 virginica のplot(青のx)
    plt.scatter(
        dat_X_std[50:100,0], dat_X_std[50:100,1],
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
        range(1, len(ada1.cost_) + 1), 
        ada1.cost_, 
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
    Plot2D.Plot2D.drawDiscriminantRegions( dat_X = dat_X_std, dat_y = dat_y, classifier = ada1 )

    # save & show
    plt.savefig("./AdalineSGD_1.png", dpi=300)
    plt.show()

    #----------------------------------------------------
    #   Lerning Process init
    #----------------------------------------------------
    ada2 = AdaLineSGD.AdaLineSGD( lRate = 0.01, numIter = 5, random_state = 1 )
    ada2.fit( X_train = dat_X_std, y_train = dat_y )
        
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
    Plot2D.Plot2D.drawDiscriminantRegions( dat_X = dat_X_std, dat_y = dat_y, classifier = ada2 )

    #----------------------------------------------------
    #   Online Learning Process
    #----------------------------------------------------
    # ストリーミングデータ (5~10) でオンライン学習
    for smIndex in range(5,100):
        print(smIndex)
        ada2.online_fit( X_train = dat_X_std[0:smIndex, :], y_train = dat_y[0:smIndex] )

    """    
    ada2.online_fit( X_train = dat_X_std[0:5, :], y_train = dat_y[0:5] )
    ada2.online_fit( X_train = dat_X_std[0:6, :], y_train = dat_y[0:6] )
    ada2.online_fit( X_train = dat_X_std[0:7, :], y_train = dat_y[0:7] )
    ada2.online_fit( X_train = dat_X_std[0:8, :], y_train = dat_y[0:8] )
    ada2.online_fit( X_train = dat_X_std[0:9, :], y_train = dat_y[0:9] )
    
    ada2.online_fit( X_train = dat_X_std[0:10, :], y_train = dat_y[0:10] )
    ada2.online_fit( X_train = dat_X_std[0:11, :], y_train = dat_y[0:11] )
    ada2.online_fit( X_train = dat_X_std[0:12, :], y_train = dat_y[0:12] )
    ada2.online_fit( X_train = dat_X_std[0:13, :], y_train = dat_y[0:13] )
    ada2.online_fit( X_train = dat_X_std[0:14, :], y_train = dat_y[0:14] )

    ada2.online_fit( X_train = dat_X_std[0:15, :], y_train = dat_y[0:15] )
    ada2.online_fit( X_train = dat_X_std[0:16, :], y_train = dat_y[0:16] )
    ada2.online_fit( X_train = dat_X_std[0:17, :], y_train = dat_y[0:17] )
    ada2.online_fit( X_train = dat_X_std[0:18, :], y_train = dat_y[0:18] )
    ada2.online_fit( X_train = dat_X_std[0:19, :], y_train = dat_y[0:19] )

    ada2.online_fit( X_train = dat_X_std[0:20, :], y_train = dat_y[0:20] )
    ada2.online_fit( X_train = dat_X_std[0:21, :], y_train = dat_y[0:21] )
    ada2.online_fit( X_train = dat_X_std[0:22, :], y_train = dat_y[0:22] )
    ada2.online_fit( X_train = dat_X_std[0:23, :], y_train = dat_y[0:23] )
    ada2.online_fit( X_train = dat_X_std[0:24, :], y_train = dat_y[0:24] )

    ada2.online_fit( X_train = dat_X_std[0:30, :], y_train = dat_y[0:30] )
    """
    #----------------------------------------------------
    #   Draw Figure 2-2
    #----------------------------------------------------
    plt.subplot(2,2,3) # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.grid(linestyle='-')
    plt.plot(
        range(1, len(ada2.cost_) + 1), 
        ada2.cost_, 
        marker = 'o',
        markersize = 0.05
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

    Plot2D.Plot2D.drawDiscriminantRegions( dat_X = dat_X_std, dat_y = dat_y, classifier = ada2 )


    # save & show
    plt.savefig("./AdalineSGD_2.png", dpi=300)
    plt.show()

if __name__ == '__main__':
     main()
