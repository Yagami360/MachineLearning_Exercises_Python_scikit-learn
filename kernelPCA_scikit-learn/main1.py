# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    print("Enter main()")
    #==========================================================================
    # カーネル主成分分析 [kernelPCA] による教師なしデータの次元削除、特徴抽出
    # scikit-learn ライブラリでのカーネル主成分分析使用
    #==========================================================================
    #====================================================
    #   Data Preprocessing（前処理）
    #====================================================
    #----------------------------------------------------
    #   read & set  data
    #----------------------------------------------------
    # 検証用サンプルデータセットの生成
    dat_X, dat_y = DataPreProcess.DataPreProcess.generateMoonsDataSet()       # 半月状のデータセット
    #dat_X, dat_y = DataPreProcess.DataPreProcess.generateCirclesDataSet()     # 同心円状のデータセット

    
    
    #========================================================================
    # Learning Process
    #========================================================================
    # scikit-learn ライブラリでの PCA
    pca1 = PCA( n_components = 2 )      # PC1, PC2
    pca2 = PCA( n_components = None)    # 主成分（固有値）解析用

    X_pca1 = pca1.fit_transform( dat_X )
    X_pca2 = pca2.fit_transform( dat_X )

    # pca2 オブジェクトの内容確認
    print( "pca2.explained_variance_ : \n", pca2.explained_variance_ )  
    print( "pca2.explained_variance_ratio_ : \n", pca2.explained_variance_ratio_ )  # 寄与率（分散比）
    print( "numpy.cumsum( pca2.explained_variance_ratio_ ) : \n", numpy.cumsum( pca2.explained_variance_ratio_ ) )  # 累積寄与率
    print( "pca2.components_ : \n", pca2.components_ )              # 主成分ベクトル（固有ベクトル）
    print( "pca2.get_covariance() : \n", pca2.get_covariance() )    # 共分散分散行列
    print( "numpy.linalg.eig( pca2.get_covariance() )[0] : \n" , numpy.linalg.eig( pca2.get_covariance() )[0] )  # 固有値のリスト

    # scikit-learn ライブラリでの kernelPCA
    scikit_kpca1 = KernelPCA( 
        n_components = 2, 
        kernel = 'rbf',         # カーネル関数として, RBF カーネルを指定
        gamma = 15 
    )
    scikit_kpca2 = KernelPCA( 
        n_components = None, 
        kernel = 'rbf',         # カーネル関数として, RBF カーネルを指定
        gamma = 15 
    )

    X_scikit_kpca1 = scikit_kpca1.fit_transform( dat_X )
    X_scikit_kpca2 = scikit_kpca2.fit_transform( dat_X )

    # scikit_kpca2 オブジェクトの内容確認
    print( "scikit_kpca2.get_params() : \n", scikit_kpca2.get_params() )
    print( "scikit_kpca2.coef0 : \n", scikit_kpca2.coef0 )

    #====================================================
    #   汎化性能の評価
    #====================================================

    #==========================================================
    #   検証用サンプルデータ（半月）での図を plot（通常のPCA）
    #==========================================================
    #------------------------------------
    # サンプルデータの散布図 plot
    #------------------------------------
    # 現在の図をクリア
    plt.clf()
    
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 3, 1)

    plt.grid()

    # サンプルデータの散布図を plot
    plt.scatter(
        dat_X[ dat_y == 0, 0 ], dat_X[ dat_y == 0, 1 ], 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        dat_X[ dat_y == 1, 0 ], dat_X[ dat_y == 1, 1 ], 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )
    plt.title( "verification data \n sklearn.datasets.make_moons() dataset" )
    plt.legend( loc = 'best' )
    #plt.tight_layout()


    #----------------------------------------
    # 変換した主成分空間での散布図 plot
    #----------------------------------------
    # x_axis = PC1, y_axis = PC2 (２次元→２次元で次元削除を行わない)

    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 3, 2)
    plt.grid()

    # サンプルデータの散布図を plot
    plt.scatter(
        X_pca1[ dat_y == 0, 0 ], X_pca1[ dat_y == 0, 1 ], 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        dat_X[ dat_y == 1, 0 ], X_pca1[ dat_y == 1, 1 ], 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )

    plt.title("transformed data (PCA) \n dimension is not deleted")
    plt.xlabel( "PC1" )
    plt.ylabel( "PC2" )
    plt.legend( loc = 'best' )
    #plt.tight_layout()


    # x_axis = PC1 (２次元→１次元で次元削除)

    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 3, 3)
    
    # サンプルデータの散布図を plot
    plt.scatter(
        X_pca1[ dat_y == 0, 0 ], numpy.zeros( (50,1) ) + 0.02, 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        dat_X[ dat_y == 1, 0 ], numpy.zeros( (50,1) ) - 0.02, 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )

    plt.title("transformed data (PCA) \n dimension is deleted")
    plt.xlabel( "PC1" )
    plt.ylim( [-1,1] )
    plt.axhline( 0.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.yticks( [] )
    plt.legend( loc = 'best' )
    #plt.tight_layout()
    
    #plt.show()
    
    #------------------------------------
    # 第 k 主成分の固有値の図 plot
    #------------------------------------
    # 現在の図をクリア
    #plt.clf()
    
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 3, 4)

    # 棒グラフ（第１主成分, 第２主成分）
    plt.bar(
        range(1, 3), numpy.linalg.eig( pca2.get_covariance() )[0], 
        alpha = 1.0, 
        align = 'center',
        label = 'Eigenvalues',
        color = "blue"
    )

    plt.axhline( 0.2, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.4, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.6, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 0.8, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.axhline( 1.0, color = 'gray', linestyle = '--', linewidth = 1 )
    
    plt.xticks( 
        range(1, 3), 
        [ "lamda_1", "lamda_2", "" ],
        rotation = 90
    )
    
    plt.title("Principal components - Eigenvalues (PCA)")
    plt.xlabel('Principal components')
    plt.ylabel('Eigenvalues')
    plt.legend( loc = 'best' )
    plt.tight_layout()
    
    #----------------------------------------
    # 第 k 主成分の寄与率＆累積寄与率の plot
    #----------------------------------------    
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 3, 5)

    # 棒グラフ（第１主成分, 第２主成分）赤棒
    plt.bar(
        range(1, 3), pca2.explained_variance_ratio_, 
        alpha = 1.0, 
        align = 'center',
        label = ' explained variance ratio(principal component 1 and 2)',
        color = "blue"
    )

    # 累積寄与率の階段グラフ
    plt.step(
        range(1, 3), numpy.cumsum( pca2.explained_variance_ratio_ ), 
        where = 'mid',
        label='cumulative proportion of the variance',
        color = "red"
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
 
    plt.xticks( range(1, 3), range(1, 3) )

    plt.title("Principal components - Proportion of the variance (PCA)")
    plt.xlabel('Principal components')
    plt.ylabel('Proportion of the variance \n individual explained variance')
    plt.legend( loc = 'best' )
    plt.tight_layout()

    plt.savefig("./kernelPCA_scikit-learn_1.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()


    #===========================================================
    #   検証用サンプルデータ（半月）での図を plot（カーネルPCA）
    #===========================================================
    #------------------------------------
    # サンプルデータの散布図 plot
    #------------------------------------
    # 現在の図をクリア
    plt.clf()
    
    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 2, 1)

    plt.grid()

    # サンプルデータの散布図を plot
    plt.scatter(
        dat_X[ dat_y == 0, 0 ], dat_X[ dat_y == 0, 1 ], 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        dat_X[ dat_y == 1, 0 ], dat_X[ dat_y == 1, 1 ], 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )
    plt.title( "verification data \n sklearn.datasets.make_moons() dataset" )
    plt.legend( loc = 'best' )
    #plt.tight_layout()


    #----------------------------------------
    # 変換した主成分空間での散布図 plot
    #----------------------------------------
    # x_axis = PC1, y_axis = PC2 (２次元→２次元で次元削除を行わない)

    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 2, 3)
    plt.grid()

    # サンプルデータの散布図を plot
    plt.scatter(
        X_scikit_kpca1[ dat_y == 0, 0 ], X_scikit_kpca1[ dat_y == 0, 1 ], 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        X_scikit_kpca1[ dat_y == 1, 0 ], X_scikit_kpca1[ dat_y == 1, 1 ], 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )

    plt.title("transformed data (RBF-kernel PCA) \n dimension is not deleted")
    plt.xlabel( "PC1" )
    plt.ylabel( "PC2" )
    plt.legend( loc = 'best' )
    #plt.tight_layout()


    # x_axis = PC1 (２次元→１次元で次元削除)

    # plt.subplot(行数, 列数, 何番目のプロットか)
    plt.subplot(2, 2, 4)
    
    # サンプルデータの散布図を plot
    plt.scatter(
        X_scikit_kpca1[ dat_y == 0, 0 ], numpy.zeros( (50,1) ) + 0.02, 
        color = 'red', 
        marker = '^', 
        label = '0',
        alpha = 0.5
    )

    plt.scatter(
        X_scikit_kpca1[ dat_y == 1, 0 ], numpy.zeros( (50,1) ) - 0.02, 
        color = 'blue', 
        marker = 'o',
        label = '1',
        alpha = 0.5
    )

    plt.title("transformed data (RBF-kernelPCA) \n dimension is deleted")
    plt.xlabel( "PC1" )
    plt.ylim( [-1,1] )
    plt.axhline( 0.0, color = 'gray', linestyle = '--', linewidth = 1 )
    plt.yticks( [] )
    plt.legend( loc = 'best' )
    #plt.tight_layout()
    
    plt.savefig("./kernelPCA_scikit-learn_2.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()
