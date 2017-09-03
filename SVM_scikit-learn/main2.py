# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
#from sklearn import datasets                            # 
#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.metrics import accuracy_score              # 
from sklearn.svm import SVC                             # 

# 自作クラス
from MLPlot import MLPlot                               # 機械学習用の図の描写をサポートする関数群からなるクラス

def main():
    #==================================================================================================
    # RBF-Kernelを使用したSVMによる非線形問題（２クラス）
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

    # 標準正規分布に従う乱数で row:200, col:2 の行列生成
    X_features = numpy.random.randn( 200, 2 )

    # X_features を XORした結果でクラス分けする
    y_labels = numpy.logical_xor( 
                (X_features[:,0] > 0),  # １列目と２列目どちらかが正と成るか？
                (X_features[:,1] > 0)
            )   
    
    y_labels = numpy.where( y_labels > 0 , 1, -1 )

    #---------------------------------------------------------------------
    # トレーニングされたモデルの性能評価を未知のデータで評価するために、
    # データセットをトレーニングデータセットとテストデータセットに分割する
    #---------------------------------------------------------------------
    # scikit-learn の cross_validation モジュールの関数 train_test_split() を用いて、70%:テストデータ, 30%:トレーニングデータに分割
    train_test = train_test_split(          # 戻り値:list
                     X_features, y_labels,  # 
                     test_size = 0.3,       # 0.0~1.0 で指定 
                     random_state = 0       # 
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
    #X_features_std = numpy.copy( X_features )                                           # ディープコピー（参照コピーではない）
    #X_features_std[:,0] = ( X_features[:,0] - X_features[:,0].mean() ) / X_features[:,0].std()  # 0列目全てにアクセス[:,0]
    #X_features_std[:,1] = ( X_features[:,1] - X_features[:,1].mean() ) / X_features[:,1].std()
    
    #====================================================
    #   Learning Process
    #====================================================
    # RBFカーネル（gamma=0.10）でのカーネルトリックを使うC-SVM（C=10）
    kernelSVM1 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 0.10,       # RFBカーネル関数のγ値
        C = 10.0,           # C-SVM の C 値
        probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )
    kernelSVM1.fit( X_train_std, y_train )

    """
    kernelSVM2 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 0.20,       # RFBカーネル関数のγ値
        C = 1               # C-SVM の C 値
    )
    kernelSVM2.fit( X_train_std, y_train )

    kernelSVM3 = SVC( 
        kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
        random_state = 0, 
        gamma = 100,        # RFBカーネル関数のγ値
        C = 1               # C-SVM の C 値
    )
    kernelSVM3.fit( X_train_std, y_train )
    """

    #====================================================
    #   汎化性能の評価
    #====================================================
    #-------------------------------------
    # サンプルデータの図示
    #-------------------------------------
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(1,2,1)

    plt.grid(linestyle='-')

    # class +1 plot(赤の□)
    plt.scatter(
        X_features[ y_labels == 1, 0 ], X_features[ y_labels == 1 , 1 ],
        color = "red",
        edgecolor = 'black',
        marker = "s",
        label = "1"
    )
    # class -1 plot(青のx)
    plt.scatter(
        X_features[ y_labels == -1, 0 ], X_features[ y_labels == -1 , 1 ],
        color = "blue",
        edgecolor = 'black',
        marker = "x",
        label = "-1"
    )

    plt.title("XOR data (generated by ramdam Normal Disuturibution)")     # title
    plt.xlim( [-3,3] )
    plt.ylim( [-3,3] )
    plt.legend(loc = "upper left")              # 凡例    
    #plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    #-------------------------------
    # 識別結果＆識別領域の表示
    #-------------------------------
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(1,2,2)

    MLPlot.drawDiscriminantRegions( 
        X_features = X_combined_std, y_labels = y_combined,
        classifier = kernelSVM1,
        list_test_idx = range( 101,150 )
    )
    plt.title("Idification Result (γ=0.1 C=10)")     # title
    plt.xlim( [-3,3] )
    plt.ylim( [-3,3] )
    plt.legend(loc = "upper left")              # 凡例    
    #plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    

    plt.savefig("./SVM_scikit-learn_3.png", dpi=300)
    plt.show()

    #-------------------------------
    # 識別率を計算＆出力
    #-------------------------------
    y_predict = kernelSVM1.predict( X_test_std )
     
    print("<テストデータの識別結果>")
    # 誤分類のサンプル数を出力
    print( "誤識別数 [Misclassified samples] : %d" % (y_test != y_predict).sum() )  # %d:10進数, string % data :文字とデータ（値）の置き換え

    # 分類の正解率を出力
    print( "正解率 [Accuracy] : %.2f" % accuracy_score(y_test, y_predict) )

    #--------------------------------------------------------------------------------------------------------
    # predict_proba() 関数を使用して、指定したサンプルのクラスの所属関係を予想
    #--------------------------------------------------------------------------------------------------------
    pre0 = kernelSVM1.predict_proba( X_test_std[0, :].reshape(1, -1) )   # 0番目のテストデータをreshap でタプル化して渡す
    pre1 = kernelSVM1.predict_proba( X_test_std[1, :].reshape(1, -1) )   # 1番目のテストデータをreshap でタプル化して渡す
    pre2 = kernelSVM1.predict_proba( X_test_std[2, :].reshape(1, -1) )   # 2番目のテストデータをreshap でタプル化して渡す
    pre3 = kernelSVM1.predict_proba( X_test_std[3, :].reshape(1, -1) )   # 3番目のテストデータをreshap でタプル化して渡す
    
    print("サンプル0の所属クラス確率 [%] :", pre0[0]*100 )
    print("サンプル1の所属クラス確率 [%] :", pre1[0]*100 )
    print("サンプル2の所属クラス確率 [%] :", pre2[0]*100 )
    print("サンプル3の所属クラス確率 [%] :", pre3[0]*100 )

    #------------------------------------------------------------------------
    # 各々のサンプルの所属クラスの図示
    #------------------------------------------------------------------------
    # 現在の図をクリア
    plt.clf()

    # 所属クラスの確率を棒グラフ表示(1,1)
    plt.subplot(2,2,1)  # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.title("Probability of class (use predict_proba method)")
    plt.xlabel("belonging class")               # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.legend(loc = "upper left")              # 凡例    

    # 棒グラフ
    plt.bar(
        left = [0,1],
        height  = pre0[0]*100,
        tick_label = ["+1","-1"]
    )             
    plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。
    
    # 所属クラスの確率を棒グラフ表示(1,2)
    plt.subplot(2,2,2)
    plt.xlabel("belonging class")               # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1],
        height  = pre0[0]*100,
        tick_label = ["+1","-1"]
    )             
    plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,3)
    plt.xlabel("belonging class)")              # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1],
        height  = pre0[0]*100,
        tick_label = ["+1","-1"]
    )             
    plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 所属クラスの確率を棒グラフ表示(2,1)
    plt.subplot(2,2,4)
    plt.xlabel("belonging class)")              # label x-axis
    plt.ylabel("probability[%]")                # label y-axis
    plt.ylim( 0,100 )                           # y軸の範囲(0~100)
    plt.bar(
        left = [0,1],
        height  = pre0[0]*100,
        tick_label = ["+1","-1"]
    )             
    plt.tight_layout()                          # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig("./SVM_scikit-learn_4.png", dpi=300)
    plt.show()

    
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

