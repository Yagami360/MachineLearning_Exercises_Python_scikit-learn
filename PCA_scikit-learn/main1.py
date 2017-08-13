# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt


# 自作クラス
import Plot2D
import DataPreProcess


def main():
    print("Enter main()")
    #=============================================================================================
    # 主成分分析 [PCA : Principal Component Analysis] による教師なしデータの次元削除、特徴抽出
    # scikit-learn ライブラリでの主成分分析不使用
    #=============================================================================================

    #====================================================
    #   Data Preprocessing（前処理）
    #====================================================
    #----------------------------------------------------
    #   read & set  data
    #----------------------------------------------------
    prePro = DataPreProcess.DataPreProcess()

    # Wine データセットの読み込み
    prePro.setDataFrameFromCsvFile( "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" )
    
    # 上記URLのWine データセットにはラベルがついてないので, 列名をセット
    prePro.setColumns( 
        [
            'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
            'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
        ] 
    )

    prePro.print("Wine データセット")

    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( 
        X_input = prePro.df_.iloc[:, 1:].values,   # iloc : 行、列を番号で指定（先頭が 0）。df_.iloc[:, 1:] = 全行、1~の全列
        y_input = prePro.df_.iloc[:, 0].values,    #
        ratio_test = 0.3
    )

    # 分割データ（トレーニングデータ、テストデータ）を出力
    print( "トレーニングデータ : \n", X_train )
    print("テストデータ : \n", X_test )
    print("トレーニング用教師データ : \n", y_train )
    print("テスト用教師データ : \n", y_test )

    # 特徴量のスケーリング（標準化 [standardization]）
    X_train_std, X_test_std \
    = DataPreProcess.DataPreProcess.standardizeTrainTest( X_train, X_test )

    # 標準化後のデータを出力
    print( "トレーニングデータ [standardized] :\n", X_train_std )
    print("テストデータ [standardized] : \n", X_test_std )    
    
    #========================================================================
    # PCAによる各主成分に対する固有値＆寄与率の算出と次元削除（特徴抽出）
    #========================================================================
    # トレーニングデータ（の転置）から共分散分散行列を作成
    Conv_mat = numpy.cov( X_test_std.T )    

    # 固有値 [eigenvalue] , 固有ベクトルの算出
    # numpylinalg.eig() により,固有分解 [eigen decomposition] を実行し,
    # 13 個の固有値とそれに対応する固有ベクトルを作成する.
    eigen_values, eigen_vecs = numpy.linalg.eig( Conv_mat )

    print( '\n固有値 [eigenvalue] \n%s' % eigen_values )
    print( '\nEigen vectors \n%s' % eigen_vecs )

    # 固有値の和をとる
    eigen_total = sum( eigen_values )
    
    # 寄与率（分散の比）[proportion of the variance] を計算（リストの内包表記）
    var_ratio = [ (ramda / eigen_total) for ramda in sorted( eigen_values, reverse = True ) ]

    print( "\n寄与率（分散の比）[proportion of the variance] \n|%-5s|" % var_ratio )

    # 累積寄与率 [Cumulative contribution rate] を計算
    cum_var_ratio = numpy.cumsum( var_ratio )

    print( "\n累積寄与率 [Cumulative contribution rate \n|%-5s|" % cum_var_ratio )

    # 特徴変換（射影行列の作成）
    # 固有値, 固有ベクトルからなるタプルのリストを作成
    eigen_pairs = [ ( numpy.abs( eigen_values[i] ), eigen_vecs[:, i] ) for i in range( len(eigen_values) ) ]

    # タプルを大きい順に並び替え
    eigen_pairs.sort( key = lambda k: k[0], reverse=True )  # ?

    # 13×2の射影行列の作成
    W_mat = numpy.hstack(
                ( eigen_pairs[0][1][:, numpy.newaxis], eigen_pairs[1][1][:, numpy.newaxis] )   # ?
            )

    print('Matrix W:\n', W_mat)

    # 作成した射影行列でトレーニングデータを変換
    X_train_pca = X_train_std.dot( W_mat )

    #====================================================
    #   汎化性能の評価
    #====================================================
    #------------------------------------
    # 第 k 主成分の固有値の図 plot
    #------------------------------------
    # 現在の図をクリア
    plt.clf()
    
    # 棒グラフ（第１主成分, 第２主成分）赤棒
    plt.bar(
        range(1, 3), eigen_values[0:2], 
        alpha = 1.0, 
        align = 'center',
        #label = 'Eigenvalues',
        color = "red"
    )

    # 棒グラフ（第３主成分, ...）青棒
    plt.bar(
        range(3, 14), eigen_values[2:13], 
        alpha = 1.0, 
        align = 'center',
        #label = 'Eigenvalues',
        color = "blue"
    )

    #plt.grid()
    plt.axhline( 1.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 2.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 3.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 4.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 5.0, color = 'gray', linestyle = '--', linewidth = 1 )
    
    plt.xticks( 
        range(1, 14), 
        [ "lamda_1", "lamda_2", "lamda_3", "lamda_4", "lamda_5" ,"lamda_6", "lamda_7", "lamda_8", "lamda_9", "lamda_10", "lamda_11", "lamda_12", "lamda_13" ],
        rotation = 90
    )

    plt.title("Principal components - Eigenvalues (PCA)")
    plt.xlabel('Principal components')
    plt.ylabel('Eigenvalues')
    plt.legend( loc = 'best' )
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./PCA_scikit-learn_1.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()


    #----------------------------------------
    # 第 k 主成分の寄与率＆累積寄与率の plot
    #----------------------------------------
    # 現在の図をクリア
    plt.clf()
    
    # 棒グラフ（第１主成分, 第２主成分）赤棒
    plt.bar(
        range(1, 3), var_ratio[0:2], 
        alpha = 1.0, 
        align = 'center',
        label = 'Eigenvalues (principal component 1 and 2)',
        color = "red"
    )

    # 棒グラフ（第３主成分, ...）青棒
    plt.bar(
        range(3, 14), var_ratio[2:13], 
        alpha = 1.0, 
        align = 'center',
        label = 'Eigenvalues (principal component 3 and so on)',
        color = "blue"
    )

    # 累積寄与率の階段グラフ
    plt.step(
        range(1, 14), cum_var_ratio, 
        where = 'mid',
        label='cumulative proportion of the variance'
    )

    plt.axhline( 0.1, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.2, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.3, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.4, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.5, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.6, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.7, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.8, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.9, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 1.0, color = 'gray', linestyle = '--', linewidth = 1 )
 
    plt.xticks( range(1, 14), range(1, 14) )

    plt.title("Principal components - Proportion of the variance (PCA)")
    plt.xlabel('Principal components')
    plt.ylabel('Proportion of the variance \n individual explained variance')
    plt.legend( loc = 'best' )
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./PCA_scikit-learn_2.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    #--------------------------------------------------------
    # 13 次元 → 2 次元に次元削除した主成分空間での散布図
    #--------------------------------------------------------
    # 現在の図をクリア
    plt.clf()
    plt.grid()

    # パレット
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(numpy.unique(y_train), colors, markers):
        plt.scatter(
            X_train_pca[y_train == l, 0], # PC1 : class l (l=1,2,3)
            X_train_pca[y_train == l, 1], # PC2 : class l (l=1,2,3)
            c = c, 
            label = l, 
            marker = m
        )
    
    plt.title("Dimension deleted Wine data (PCA) \n 13×178 dim → 2×124 dim [dimension / feature extraction]")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # 図の保存＆表示
    plt.savefig("./PCA_scikit-learn_3.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()
