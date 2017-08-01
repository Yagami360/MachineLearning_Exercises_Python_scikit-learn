# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz


class DecisionTree(object):
    """
    決定木 [DecisionTree] を表すクラス
    scikit-learn ライブラリの sklearn.tree モジュールにある DecisionTreeClassifier クラスのラッパークラス

    [public] 
        tree_ : DecisionTreeClassifier クラスのオブジェクト
    """
    def __init__( self, purity = 'gini', max_depth = 3, random_state = 0 ):
        self.tree_ = DecisionTreeClassifier( 
            criterion = purity, 
            max_depth = max_depth, 
            random_state = random_state
        )

        return

    @staticmethod
    def clacNodeError( p ):
        """
        ノードの誤り率を計算する.
        [Input]
            p : float
                確率値（0~1）
        [Output]
            入力 p に対する計算結果
        """
        value = 1 - numpy.max( [p, 1-p] )
        return value

    @staticmethod
    def clacGineIndex( p ):
        value = p * (1 - p) + (1 - p) * ( 1 - (1 - p) )
        return value

    @staticmethod
    def calcCrossEntropy( p ):
        # ２クラスのバイナリーなのでlog の底が 2
        value = - p * numpy.log2(p) - (1 - p) * numpy.log2( (1 - p) )
        
        return value


    def plotNodeErrorFunction( self, figure, axis, x = numpy.arange( 0.0, 1.0, 0.01 ) ):
        """
         ノードの誤り率を表す関数を描写
         [Input]
            figure : Figure クラスのオブジェクト
                plt.figure() で作成されるオブジェクト
            axis : Axis クラスのオブジェクト
                figure.subplot() で作成されるオブジェクト
            x : darry
                確率の値 0~1 からなる配列（x軸の値に相当）
        """
        # 配列 x の値を元にノードの誤り率を計算し, errors リストに格納
        errors = []
        for p in x:
            error = self.clacNodeError( p )
            errors.append( error )
            
        # 図の作図
        axis.plot(
            x, errors,
            label = "Misclassification Error",
            linestyle = "-.",
            lw = 2,
            color = "cyan"
        )

        #
        axis.axvline( x = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 1.0, linewidth = 0.5, color = 'k', linestyle = '--' )

        plt.ylim( [0, 1.1] )
        plt.xlabel( "p (i=1)" )
        plt.ylabel( "green" )

        return

    def plotCrossEntropyFunction( self, figure, axis, x = numpy.arange( 0.0, 1.0, 0.01 ) ):
        # 配列 x の値を元にノードの交差エントロピーを計算し, リストに格納
        ents = []
        for p in x:
            if ( p == 0.01 ):
                ents.append( None )
            else:
                ent = self.calcCrossEntropy( p )
                #print("p,ent", p, ent)
                ents.append( ent )
        
        # 図の作図
        axis.plot(
            x, ents,
            label = "Cross Entropy",
            linestyle = "--",
            lw = 2,
            color = "blue"
        )

        #
        axis.axvline( x = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 1.0, linewidth = 0.5, color = 'k', linestyle = '--' )

        plt.ylim( [0, 1.1] )
        plt.xlabel( "p (i=1)" )
        plt.ylabel( "Purity" )

        return

    def plotGiniIndexFunction( self, figure, axis, x = numpy.arange( 0.0, 1.0, 0.01 ) ):
        # 配列 x の値を元にジニ係数を計算し, リストに格納
        gines = []
        for p in x:
            gine = self.clacGineIndex( p )
            gines.append( gine )
        
        # 図の作図
        axis.plot(
            x, gines,
            label = "Gini Index",
            linestyle = "-",
            lw = 2,
            color = "red"
        )

        #
        axis.axvline( x = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 0.5, linewidth = 0.5, color = 'k', linestyle = '--' )
        axis.axhline( y = 1.0, linewidth = 0.5, color = 'k', linestyle = '--' )

        plt.ylim( [0, 1.1] )
        plt.xlabel( "p (i=1)" )
        plt.ylabel( "Purity" )