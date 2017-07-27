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

# 自作クラス
import Plot2D
import LogisticRegression

def main():

    #==================================================================================
    #   ロジスティクス回帰処理
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
    #dat_X_std = numpy.copy(dat_X)                                           # ディープコピー（参照コピーではない）
    #dat_X_std[:,0] = ( dat_X[:,0] - dat_X[:,0].mean() ) / dat_X[:,0].std()  # 0列目全てにアクセス[:,0]
    #dat_X_std[:,1] = ( dat_X[:,1] - dat_X[:,1].mean() ) / dat_X[:,1].std()

    #====================================================
    #   Learning Process (& 説明用の図のpolt)
    #====================================================
    logReg = LogisticRegression.LogisticRegression()
    
    #plt.subplot(2,1,1)              # plt.subplot(行数, 列数, 何番目のプロットか)
    logReg.plotSigmoidFunction()
    plt.savefig("./LogisticRegression_scikit-learn_1.png", dpi=300)
    #plt.subplot(2,1,2)              # plt.subplot(行数, 列数, 何番目のプロットか)
    logReg.plotCostFunction()
    
    plt.savefig("./LogisticRegression_scikit-learn_2.png", dpi=300)
    #plt.show()

    logReg.logReg_.fit( X_train_std, y_train )
    

    #====================================================
    #   汎化性能の評価
    #====================================================
    #-------------------------------
    # 識別結果＆識別領域の表示
    #-------------------------------
    Plot2D.Plot2D.drawDiscriminantRegions( 
        dat_X = X_combined_std, dat_y = y_combined,
        classifier = logReg.logReg_ ,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result")         #
    plt.xlabel("sepal length [Normalized]") # label x-axis
    plt.ylabel("petal length [Normalized]") # label y-axis
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    plt.savefig("./LogisticRegression_scikit-learn_3.png", dpi=300)
    #plt.show()

    #-------------------------------
    # 識別率を計算＆出力
    #-------------------------------
    y_predict = logReg.logReg_.predict( X_test_std )

    # 誤分類のサンプル数を出力
    print( 'Misclassified samples: %d' % (y_test != y_predict).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え

    # 分類の正解率を出力
    print( 'Accuracy: %.2f' % accuracy_score(y_test, y_predict) )

    #--------------------------------------------------------------------------------------------------------
    # predict_proba() 関数を使用して、指定したサンプルのクラスの所属関係を予想
    # 戻り値は、サンプルが Iris-Setosa, Iris-Versicolor, Iris-Virginica に所属する確率をこの順で表している.
    #--------------------------------------------------------------------------------------------------------
    pre0 = logReg.logReg_.predict_proba( X_test_std[0, :].reshape(1, -1) )   # 0番目のテストデータをreshap でタプル化して渡す
    pre1 = logReg.logReg_.predict_proba( X_test_std[1, :].reshape(1, -1) )   # 1番目のテストデータをreshap でタプル化して渡す
    pre2 = logReg.logReg_.predict_proba( X_test_std[2, :].reshape(1, -1) )   # 2番目のテストデータをreshap でタプル化して渡す
    pre3 = logReg.logReg_.predict_proba( X_test_std[3, :].reshape(1, -1) )   # 3番目のテストデータをreshap でタプル化して渡す
    
    #print("predict:", pre0)
    #print("predict[0]:", pre0[0])
    #print("predict[0,0]:", pre0[0,0]*100)
    #print("predict[0,1]:", pre0[0,1]*100)
    #print("predict[0,2]:", pre0[0,2]*100)

    #------------------------------------------------------------------------
    # 各々のサンプルの所属クラスの図示 ["Setosa","Versicolor","Virginica"]
    #------------------------------------------------------------------------
    # 現在の図をクリア
    plt.clf()

    # 所属クラスの確率を棒グラフ表示(1,1)
    plt.subplot(2,2,1)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.title("Probability of class (use predict_proba mesod)")
    plt.xlabel("Varieties (Belonging class) [Setosa,Versicolor,Virginica]")    # label x-axis
    plt.ylabel("probability[%]")        # label y-axis
    plt.ylim( 0,100 )                   # y軸の範囲(0~100)
    plt.legend(loc = "upper left")      # 凡例    

    # 棒グラフ
    plt.bar(
        left = [0,1,2],
        height  = pre0[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    # 所属クラスの確率を棒グラフ表示(1,2)
    plt.subplot(2,2,2)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.ylim( 0,100 )                   # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre1[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,3)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.ylim( 0,100 )                   # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre2[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,4)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.ylim( 0,100 )                   # y軸の範囲(0~100)
    plt.bar(
        left = [0,1,2],
        height  = pre3[0]*100,
        tick_label = ["Setosa","Versicolor","Virginica"]
    )             
    plt.tight_layout()                  # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig("./LogisticRegression_scikit-learn_4.png", dpi=300)
    #plt.show()


    #==================================================================================================
    # ロジスティクス回帰の逆正則化パラメータ C の値と正則化の強さの関係
    # ロジスティクス回帰における、正則化による過学習への対応
    #==================================================================================================
    print("Start Process2")
    weights0 = []    # 重み係数の空リスト（Setosa）を生成
    weights1 = []    # 重み係数の空リスト（Versicolor）を生成
    weights2 = []    # 重み係数の空リスト（Virginica）を生成
    paramesC = []    # 逆正則化パラメータの空リストを生成

    # 10個の逆正則化パラメータ C（C=10^-5,C=10^-4, ... C=10, ... , C=10^5 ）に関して、LogisiticReegression オブジェクトを作成し、
    # それぞれのモデルを学習データで学習
    for c in numpy.arange(-5, 5):
        logReg_tmp = LogisticRegression.LogisticRegression( paramC = 10**c )
        logReg_tmp.logReg_.fit( X_train_std, y_train )

        # 重み係数リストに学習後の重み係数を格納
        weights0.append( logReg_tmp.logReg_.coef_[0] )   # coef_[0] : petal length, petal weightの重み (Setosa)
        weights1.append( logReg_tmp.logReg_.coef_[1] )   # coef_[1] : petal length, petal weightの重み (Versicolor)
        weights2.append( logReg_tmp.logReg_.coef_[2] )   # coef_[2] : petal length, petal weightの重み (Virginica)
        
        #print("weight0:", logReg_tmp.logReg_.coef_[0] )
        #print("weight1:", logReg_tmp.logReg_.coef_[1] )
        #print("weight2:", logReg_tmp.logReg_.coef_[2] )

        # 逆正則化パラメータリストにモデルの C 値を格納
        paramesC.append( 10**c )

    # 重み係数リストを numpy 配列に変換
    weights0 = numpy.array( weights0 )
    weights1 = numpy.array( weights1 )
    weights2 = numpy.array( weights2 )
    
    #----------------------------
    # 図の plot
    #----------------------------
    # 現在の図をクリア
    plt.clf()
    
    # Setosa
    plt.plot(
        paramesC, weights0[:, 0],            # １つ目(petal length)の重み係数 w00 
        label='petal length (Setosa)',
        color = "red",
    )
    plt.plot(
        paramesC, weights0[:, 1],            # ２つ目 (petal weight) の重み係数 w01
        linestyle = '--',
        label = 'petal width (Setosa)',
        color = "red"
    )
    # Versicolor
    plt.plot(
        paramesC, weights1[:, 0],            # １つ目(petal length)の重み係数 w10 
        label='petal length (Versicolor)',
        color = "blue"
    )
    plt.plot(
        paramesC, weights1[:, 1],            # ２つ目 (petal weight) の重み係数 w11
        linestyle = '--',
        label = 'petal width (Versicolor)',
        color = "blue"
    )
    # Virginica 
    plt.plot(
        paramesC, weights2[:, 0],            # １つ目(petal length)の重み係数 w20 
        label='petal length (Virginica)',
        color = "green"
    )
    plt.plot(
        paramesC, weights2[:, 1],            # ２つ目 (petal weight) の重み係数 w21
        linestyle = '--',
        label = 'petal width (Virginica)',
        color = "green"
    )

    plt.ylabel('weight coefficient')
    plt.xlabel('C [reverse regularization parameter](=1/λ)')
    plt.legend(loc = 'upper left')
    plt.xscale('log')   # x軸を対数スケール
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./LogisticRegression_scikit-learn_5.png", dpi=300)
    plt.show()
    
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

