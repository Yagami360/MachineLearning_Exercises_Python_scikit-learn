# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

"""
    更新情報
    [17/08/24] : 新規作成

"""

import numpy
import pandas
import matplotlib.pyplot as plt

from scipy.misc import comb             # 組み合わせの計算 
import math                             #

import operator                         # ?

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.preprocessing import LabelEncoder      # 
 
from sklearn.externals import six                   # ?           
from sklearn.base import clone                      # ?
from sklearn.pipeline import _name_estimators       # ?


class EnsembleLearningClassifier( BaseEstimator, ClassifierMixin ):
    """
    アンサンブルモデルの識別器 classifier の自作クラス.
    scikit-learn ライブラリの推定器 estimator の基本クラス BaseEstimator, ClassifierMixin を継承している.
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        
        classifiers : list <classifier オブジェクト>        
            


        vote_method : str

        weights : list <float>

        named_classifier :

        ---------------------------------

        classifiers_ :


        encoder_ : 

        classes_ :



    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
        __classifiers : list <classifier オブジェクト>
            分類器のクラスのオブジェクトのリスト

        __fitted_classifiers : list <classifier オブジェクト>
            fitting 処理後の分類器のクラスのオブジェクトのリスト

        __n_classifier : int
            アンサンブル分類器を構成する分類器の個数

        __class_labels : list <str>
            __classifiers のラベル名のリスト
            
        __weights : list <float>
            各分類器の対する重みの値のリスト

        __vote_method : str ( "majority_vote" or "probability_vote" )
            アンサンブルによる最終的な判断判断手法
            "majority_vote"    : 弱識別器の多数決で決定する.多数決方式 (＝クラスラベルの argmax() 結果）
            "probability_vote" : 弱識別器の重み付け結果で決定する.（＝クラスの所属確率の argmax() 結果）

        __encoder : sklearn.preprocessing.LabelEncoder のオブジェクト
        

        __named_classifiers : dict
            
    """
    
    def __init__( self , classifiers, class_labels = [], vote_method = "majority_vote", weights = None ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        引数と同名のオブジェクトの属性を設定する必要あり（上位クラスの BaseEstima仕様）
        
        [Input]
            classifiers_ : list
                分類器のクラスのオブジェクトのリスト
                

        """
        self.classifiers = classifiers
        
        self.class_labels = class_labels
        self.class_labels_ = class_labels

        self.weights = weights

        if classifiers != None:
            self.__n_classifier = len( classifiers )
        else:
            self.__n_classifier = 0

        self.vote_method = vote_method

        # ?
        if classifiers != None:
            self.named_classifiers = { key: value 
                                         for key, value in _name_estimators(classifiers) }
        else:
            self.named_classifiers = {}

        return

    def print( self, str = "" ):
        """

        """
        print("\n")
        print("-------------------------------------------------------------------")
        print( str )
        print( "\n[Attributes]" )

        print( "classifiers :" )
        for clf in self.classifiers:
            print( "    ", clf )

        print( "__n_classifier : ", self.__n_classifier )
        print( "class_labels : ", self.class_labels )
        print( "weights : ", self.weights )
        print( "vote_method : ", self.vote_method )

        print( "\n[self]\n", self )
        print("-------------------------------------------------------------------")
        
        return

    def fit( self, X_train, y_train ):
        """
        識別器に対し, 指定されたデータで fitting を行う関数
        scikit-learn ライブラリの識別器 : classifiler, 推定器 : estimator が持つ共通関数

        [Input]
            X_train : numpy.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）

            y_train : numpy.ndarray ( shape = [n_samples] )
                トレーニングデータ用のクラスラベル（教師データ）のリスト

        [Output]
            self : 自身のオブジェクト

        """
        # ? LabelEncoder クラスを使用して, クラスラベルが 0 から始まるようにする.
        # ? これは, self.predict() 関数内の numpy.argmax() 関数呼び出し時に重要となるためである.
        self.encoder_ = LabelEncoder()
        self.encoder_.fit( y_train )
        self.classes = self.encoder_.classes_

        # public アクセス可能な分類器のリストを初期化
        self.classifiers_ = []

        # self.classifiers に設定されている分類器のクローン clone(clf) で fitting し, 
        # public アクセス可能な classifiers_ に格納
        for clf in self.classifiers:

            # clone() : 同じパラメータの 推定器を生成
            fitted_clf = clone(clf).fit( X_train, self.encoder_.transform(y_train) )
            self.classifiers_.append( fitted_clf )

        return self


    def predict( self, X_features ):
        """
        識別器に対し, fitting された結果を元に, クラスラベルの予想値を返す関数

        [Input]
            X_features : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        [Output]
            vote_results : numpy.ndaary ( shape = [n_samples] )
                予想結果（クラスラベル）
        """
        # 初期化
        #vote_results = []

        #------------------------------------------------------------------------------------------------------
        # アンサンブルの最終決定方式 __vote_method が, 各弱識別器の重み付け方式 "probability_vote" のケース
        #------------------------------------------------------------------------------------------------------
        if self.vote_method == "probability_vote":
            # numpy.argmax() : 指定した配列の最大要素のインデックスを返す
            # axis : 最大値を読み取る軸の方向 ( axis = 1 : shape が２次元配列 行方向)
            vote_results = numpy.argmax( self.predict_proba(X_features), axis = 1 )

        #------------------------------------------------------------------------------------------------------        
        # アンサンブルの最終決定方式 __vote_method が, 多数決方式 "majority_vote" のケース
        #------------------------------------------------------------------------------------------------------
        else:
            # 各弱識別器 clf の predict() 結果を predictions (list) に格納
            #predictions = [ clf.predict(X_features) for clf in self.__fitted_classifiers ]
            #print( "EnsembleLearningClassifier.fit() { predictions } : \n", predictions)

            # ? predictions を 転置し, 行と列 (shape) を反転
            # numpy.asarray() :  np.array とほとんど同じように使えるが, 引数に ndarray を渡した場合は配列のコピーをせずに引数をそのまま返す。
            # predictions = numpy.asarray( predictions ).T
            #print( "EnsembleLearningClassifier.fit() { numpy.asarray(predictions).T } : \n", predictions)

            predictions = numpy.asarray( [ clf.predict(X_features) for clf in self.classifiers_ ] ).T

            # ? 各サンプルの所属クラス確率に重み付けで足し合わせた結果が最大となるようにし、列番号を返すようにする.
            # この処理を numpy.apply_along_axis() 関数を使用して実装
            # numpy.apply_along_axis() : Apply a function to 1-D slices along the given axis.
            # Execute func1d(a, *args) where func1d operates on 1-D arrays and a is a 1-D slice of arr along axis.
            vote_results = numpy.apply_along_axis(
                               lambda x :                                                       # func1d : function
                               numpy.argmax( numpy.bincount( x, weights = self.weights ) ),     # 
                               axis = 1,                                                        #
                               arr = predictions                                                # ndarray : Input array
                           )

        # ? vote_results を LabelEncoder で逆行列化して, shape を反転
        #print( "EnsembleLearningClassifier.fit() { vote_results } : \n", vote_results )
        vote_results = self.encoder_.inverse_transform( vote_results )
        #print( "EnsembleLearningClassifier.fit() {  self.__encoder.inverse_transform( vote_results ) } : \n", vote_results )

        return vote_results


    def predict_proba( self, X_features ):
        """
        識別器に対し, fitting された結果を元に, クラスの所属確率の予想値を返す関数

        [Input]
            X_features : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列

        [Output]
            ave_probas : numpy.nadarry ( shape = [n_samples, n_classes] )
                各サンプルの所属クラス確率に重み付けした結果の平均確率
        """
        # 各弱識別器 clf の predict_prpba() 結果を predictions (list) に格納
        #predict_probas = [ clf.predict_proba(X_features) for clf in self.__fitted_classifiers ]
        #print( "EnsembleLearningClassifier.predict_proba() { predict_probas } : \n", predict_probas )
        predict_probas = numpy.asarray( [ clf.predict_proba(X_features) for clf in self.classifiers_ ] )

        # 平均化
        ave_probas = numpy.average( predict_probas, axis = 0, weights = self.weights )
        #print( "EnsembleLearningClassifier.predict_proba() { ave_probas } : \n", ave_probas )

        return ave_probas


    def get_params( self, deep = True ):
        """
        親クラス BaseEstimator の関数 get_params() をオーバーライド
        
        """
        if deep == False:
            # ?
            return super( EnsembleLearningClassifier, self ).get_params( deep = False )
        else:
            # ? キューを "分類器の名前__パラメータ名" ,
            # バリューをパラメータ値とするディクショナリを生成
            out = self.named_classifiers.copy()     # named_classifiers.copy() : 

            # ? six.iteritems() : 
            for name, step in six.iteritems( self.named_classifiers ):
                for key, value in six.iteritems( step.get_params(deep=True) ):
                     out['%s__%s' % (name, key)] = value

            return out

    def get_classiflers( self ):
        """
        """
        return self.classifiers

    def get_class_labels( self ):
        """
        """
        return self.class_labels

    def calcEnsenbleError( self, error ):
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
                          self.calcEnsenbleError( error )
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
