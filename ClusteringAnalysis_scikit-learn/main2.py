# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets
from sklearn.datasets import make_blobs     # クラスタリングのためのガウス分布を生成

from sklearn.cluster import KMeans          # 


# 自作クラス
from EnsembleModelClassifier import EnsembleModelClassifier
from MLPreProcess import MLPreProcess
from MLPlot import MLPlot 

def main():
    """
    クラスター分析.
    k-mean 法とエルボー法を用いた最適なクラスター数
    """
    print("Enter main()")

    #---------------------------------------------------------------
    # データの読み込み
    #---------------------------------------------------------------
    # make_blobs() 関数でクラスタリングのためのガウス分布を生成
    X_features, y_labels = make_blobs(
                               n_samples = 500,     # サンプル数
                               n_features = 2,      # 特徴数
                               centers = 5,         # クラスターの個数 
                               cluster_std = 0.5,   # クラスター内での標準偏差
                               shuffle = True,      # サンプルをシャッフルするか否か
                               random_state = 0     # 乱数生成器の状態 ( If int, random_state is the seed used by the random number generator )
                           )
    
    #print( "X_features : \n", X_features )
    #print( "y_labels : \n", y_labels )

    # k-means
    kmeans = KMeans(
                n_clusters = 5,     # cluster の個数（＝ centroid の個数）
                init = 'random',    # centroid の初期値をランダムに設定
                n_init = 10,        # ? 異なるセントロイドの seed を使用した, k-means アルゴリズムの実行回数
                                    # cluster の個数の都度, 異なるランダムな初期値を使用して, k-means 法によるクラスタリングを 10 回行う.
                                    # The final results will be the best output of n_init consecutive runs in terms of inertia.
                max_iter = 300,     # １回の k-means アルゴリズム内部の最大イテレーション回数
                tol = 1e-04,        # 収束と判定する為の相対的な許容誤差値
                random_state = 0,   # セントロイドの初期化に用いる乱数生成器の状態
                n_jobs = 1          # CPU の並列処理数
            )

    # 指定した特徴行列 X_features のクラスターのセントロイドの計算＆各サンプルのインデックスを予想して返す.
    y_kmeans = kmeans.fit_predict( X_features )

    print( "kmeans.cluster_centers_ :\n", kmeans.cluster_centers_ )
    print( "kmeans.labels_ :\n", kmeans.labels_ )
    print( "y_kmeans :\n", y_kmeans )
    print( "Distortion (SSE) : %.2f" % kmeans.inertia_)

    #---------------------------------------------------------------
    # エルボー図の描写
    #---------------------------------------------------------------
    n_max_clusters = 15     # 確認するクラスター数の最大数
    distortions = []        # SSE
    
    # クラスター数 1 ~ n_max_clusters でループ
    for idx in range(1, n_max_clusters):
        kmeans_pp = KMeans(
                        n_clusters = idx,         # cluster の個数（＝ centroid の個数）
                        init = 'k-means++',       # k-means++ 法を使用
                        n_init = 10,              # 異なるセントロイドの seed を使用した, k-means アルゴリズムの実行回数
                                                  # cluster の個数の都度, 異なるランダムな初期値を使用して, k-means 法によるクラスタリングを 10 回行う.
                        max_iter = 300,           # 1 回の k-means アルゴリズム内部の最大イテレーション回数
                        random_state = 0          # 最適なクラスタ数を調べたいので, 乱数化しない
                    )

        kmeans_pp.fit( X_features )               # 特徴行列で fitting
        distortions.append( kmeans_pp.inertia_ )  # SSE のリストの最後尾に追加
        print( kmeans_pp.inertia_ )

    # distortions [SSE] のリストを描写
    plt.plot(
        range(1, n_max_clusters), distortions, 
        marker = 'o'
    )

    plt.title( "elbow method" )
    plt.xlabel( "Number of clusters" )
    plt.ylabel( "Distortion [SSE]" )
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = 'ClutterringAnalysis_scikit-learn_2-1.png' )
    plt.show()

    print("Finish main()")
    return

if __name__ == '__main__':
     main()