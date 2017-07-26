# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt
import sklearn.linear_model                             # クラス名の衝突を避けるため
#from sklearn.linear_model import LogisticRegression    # ↑


class LogisticRegression(object):
    """
        scikit-learn ライブラリでのロジスティクス回帰のラッパークラス

        [public] 
            logReg_ : sklearn.linear_model の LogisticRegression クラスのオブジェクト
    """

    def __init__( self ):
        """ """
        #  sklearn.linear_model の LogisticRegression クラスのオブジェクト生成
        self.logReg_ = sklearn.linear_model.LogisticRegression( C = 100, random_state = 0 )
        
        return

    def sigmoidFunction( self, z ):
        """
        シグモイド関数の値を求める 
        [input]
            z : numppy.ndarry
        [Output]
            引数のリスト z に対応したシグモイド関数値のリスト value
        """
        value = 1.0 / ( 1.0 + numpy.exp(-z) )
        return value

    def costFunctionClass0( self, z ):
        """
        クラス０に属しているサンプルのコスト関数の値を求める
        [input]
            z : numppy.ndarry
        [Output]
            引数のリスト z に対応したコスト関数値のリスト value        
        """
        value = -numpy.log( 1- self.sigmoidFunction(z) )
        return value

    def costFunctionClass1( self, z ):
        """
        クラス１に属しているサンプルのコスト関数の値を求める
        [input]
            z : numppy.ndarry
        [Output]
            引数のリスト z に対応したコスト関数値のリスト value        
        """
        value = -numpy.log( self.sigmoidFunction(z) )
        return value

    def plotSigmoidFunction( self, z = numpy.arange(-7, 7, 0.1), xlabel = 'z', ylabel = '$\phi (z)$', isGrid = True ):
        """ 
        シグモイド関数を図に描写する

        """
        phi_z = self.sigmoidFunction( z )
        
        """
        # 描写用のオブジェクトを作成（オブジェクト指向的な描写処理）
        fig = plt.figure()              # Figure クラスのオブジェクトを生成
        axis = fig.add_subplot(1,1,1)   # Axes クラスのオブジェクトを生成
        
        # *オブジェクトの確認
        print("type(fig): {}".format( type(fig) ) )
        print("type(axis): {}".format(type(axis)))
        
        axis.plot( z, phi_z)

        axis.set_ylim( -0.1, 1.1)
        axis.set_yticks( [0.0, 0.5, 1.0] )  # y軸の目盛り
        axis.yaxis.grid( True )             # y軸の水平グリッド線
        axis.axvline( 0.0, color='k' )      # z=0 の垂直線
        axis.set_xlabel( xlabel )
        axis.set_ylabel( ylabel )
        fig.tight_layout()
        """
        plt.clf()                      # 現在の図をクリア
        plt.plot( z, phi_z )

        plt.ylim( -0.1, 1.1 )
        plt.axvline( 0.0, color='k' )  # z=0 の垂直線
        plt.xlabel( xlabel )
        plt.ylabel( ylabel )

        plt.yticks( [0.0, 0.5, 1.0] )  # y軸の目盛り追加
        axis = plt.gca()               # Axes クラスのオブジェクト生成 
        axis.yaxis.grid(True)          # y軸の目盛りに合わせた水平グリッド線

        plt.tight_layout()
        #plt.show()

        return

    def plotCostFunction( self ):
        z = numpy.arange( -10, 10, 0.1 )
        phi_z = self.sigmoidFunction(z)

        #cost0 = [cost_0(x) for x in z]      # 書式: z の要素インデックス x で抽出し、この x で呼び出したコスト関数の値を配列の要素とする
        
        cost0 = []
        cost1 = []
        for zi in z:
            cost0.append( self.costFunctionClass0(zi) )
            cost1.append( self.costFunctionClass1(zi) )
        
        # plot figure
        plt.clf()    # 現在の図をクリア
        plt.plot(
            phi_z, cost0, 
            linestyle='--', 
            label='J(w) : cross-entropy error function (class label:0)'
        )

        plt.plot(
            phi_z, cost1, 
            linestyle='-', 
            label='J(w) : cross-entropy error function (class label:1)'
        )
        plt.ylim(0.0, 5.1)
        plt.xlim([0, 1])
        plt.xlabel('$\phi$(z)')
        plt.ylabel('J(w) : cross-entropy error function (cost)')
        plt.legend(loc='best')
        plt.tight_layout()
        
        #plt.show()
        
        return