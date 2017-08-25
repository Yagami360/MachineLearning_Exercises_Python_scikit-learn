# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

"""
    更新情報
    [17/08/24] : 新規作成

"""

import numpy
import pandas
import matplotlib.pyplot as plt

from scipy.misc import comb             # 
import math

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator      # 推定器 Estimator の上位クラス
from sklearn.base import ClassifierMixin    

class EnsembleLearningClassifier( BaseEstimator, ClassifierMixin ):
    """
    アンサンブル学習による識別器 : classifier の自作クラス.
    scikit-learn ライブラリの推定器 : estimator の基本クラス BaseEstimator, ClassifierMixin を継承している.
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
        __n_classifier : int
            アンサンブル分類器を構成する分類器の個数

    """


    def __init__( self , n_classifier ):
        """
        コンストラクタ（厳密にはイニシャライザ）

        [Input]
            n_classifier : int
                アンサンブル分類器を構成する分類器の個数
        """
        self.__n_classifier = n_classifier
        
        return

    def fit():
        """
        識別器に対し, 指定されたデータで fitting を行う関数
        scikit-learn ライブラリの識別器 : classifiler, 推定器 : estimator が持つ共通関数

        [Input]

        [Output]

        """
        return


    def predict():
        """
        識別器に対し, fitting された結果を元に, 予想値を返す関数

        [Input]

        [Output]

        """
        return

    def CalcplotEnsenbleError( self, error ):
        """
        ２項分布（多数決）に基づいた, アンサンブルの誤分類率を計算する.
        P_ensemble = ∑ n_C_k * e^k * (1-e)^n-k

        [Input]
            error : float

        [Output]
            アンサンブルの誤分類率 : float
        """
        # 組み合わせ n_C_k の 最初の k 値を算出
        k_start = int( math.ceil( self.__n_classifier / 2.0 ) )  # math.ceil() : 引数の値以上の最小の整数を返す

        # n_C_k * e^k * (1-e)^n-k 部分を計算（リストの内含的記述）
        probes = [ 
                     comb( self.__n_classifier, k ) * (error**k) * (1-error)**(self.__n_classifier - k)  # comb() : 組み合わせ
                     for k in range( k_start, self.__n_classifier + 1 ) 
                 ]

        # ∑をとる
        p_ensemble = sum(probes)

        return p_ensemble
        
    def plotEnsenbleErrorAndBaseError( self, errors = numpy.arange( 0.0 , 1.01, 0.01 ) ):
        """
        ２項分布（多数決）に基づいたアンサンブルの誤分類率と,通常の（アンサンブルでない）誤分類の図を描写する.

        [Input]
            errors : numpy.ndarray
                誤分類率のリスト（x軸に対応）
        """
        # Ensenble Error rate を計算する
        p_ensembles = [
                          self.CalcplotEnsenbleError( error )
                          for error in errors
                      ]
        
        # アンサンブルの誤分類を plot
        plt.plot(
            errors, p_ensembles, 
            label = "Ensemble error", 
            linewidth = 2
        )

        # 通常の（アンサンブルでない）誤分類を plot
        plt.plot(
            errors, errors,
            linestyle='--',
            label = "Base error", 
            linewidth = 2
        )

        # 当て推量 (eror = 0.5) の線の plot
        plt.axvline( x = 0.5, linewidth = 1.0, color = 'k', linestyle = '--' )
        plt.axhline( y = 0.5, linewidth = 1.0, color = 'k', linestyle = '--' )
        
        #
        plt.title( "Base/Ensemble error \n number of classifiler = %d" % self.__n_classifier )
        plt.xlabel( "Base error" )
        plt.ylabel( "Base/Ensemble error" )

        plt.legend( loc = "best" )
        #plt.grid()
        plt.tight_layout()

        return
