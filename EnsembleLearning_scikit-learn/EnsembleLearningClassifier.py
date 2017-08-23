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
from sklearn.base import BaseEstimator  # 推定器 Estimator の上位クラス


class EnsembleLearningClassifier( BaseEstimator ):
    """
    アンサンブル学習による識別器 : classifier の自作クラス.
    scikit-learn ライブラリの推定器 : estimator の基本クラス BaseEstimator を継承している.
    
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

    def plotEnsenbleErrorAndBaseError():
        """

        [Output]
            figure : matplotlib.figure クラスのオブジェクト
                描画される部品を納めるコンテナクラス ( Artist の派生クラス )
        """
        # Ensenble Error rate を計算する


        # Figure クラスのオブジェクト生成
        figure = plt.figure()


        return figure
