# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy

# Data Frame & IO 関連
import pandas
from io import StringIO

# scikit-learn ライブラリ関連
from sklearn import datasets                            # 
#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.metrics import accuracy_score              # 
from sklearn.preprocessing import Imputer               # 


class DataPreProcess( object ):
    """
    機械学習用のデータの前処理を行うクラス
    データフレームとして, pandas DataFrame のオブジェクトを持つ。（コンポジション：集約）
    sklearn.preprocessing モジュールのラッパークラス
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        df_ : pandas DataFrame のオブジェクト（データフレーム）
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self ):
        """
        コンストラクタ
        """
        self.df_ = pandas.DataFrame()
        return

    def print( self ):
        print( self )
        print( self.df_ )
        return

    def setDataFrame( self, dataFrame ):
        """
        [Input]
            dataFrame : list

        """
        self.df_ = pandas.DataFrame( dataFrame )

        return


    def setDataFrameFromCsvData( self, csv_data ):

        # read_csv() 関数を用いて, csv フォーマットのデータを pandas DataFrame オブジェクトに変換して読み込む.
        self.df_ = pandas.read_csv( StringIO( csv_data ) )
        return

    def setDataFrameFromCsvFile( self, csv_fileName ):
        return

    def getNumpyArray( self ):
        """
        pandas Data Frame オブジェクトを Numpy 配列にして返す
        """
        values = self.df_.values     # pandas DataFrame の value 属性
        return values

    #---------------------------------------------------------
    # 欠損値の処理を行う関数群
    #---------------------------------------------------------
    def meanImputationNaN( self, axis = 0 ):
        """
        欠損値 [NaN] を平均値で保管する
        [Input]
            axis : int
                0 : NaN を列の平均値で補完
                1 : NaN を行の平均値で補完
        """
        imputer = Imputer( 
                      missing_values = 'NaN', 
                      strategy = 'mean', 
                      axis = axis       # 0 : 列の平均値, 1 : 行の平均値
                  )
        
        imputer.fit( self.df_ )         # self.df_ は１次配列に変換されることに注意

        self.df_ = imputer.transform( self.df_ )

        return
    
    #---------------------------------------------------------
    # カテゴリデータの処理を行う関数群
    #---------------------------------------------------------
    def setColumns( self, columns ):
        """
        データフレームにコラム（列）を設定する。
        """
        self.df_.columns = columns
        
        return
    
    def MappingOrdinalFeatures( self, key, input_dict ):
        """
        順序特徴量のマッピング（整数への変換）
        
        [Input]
            key : string
                順序特徴量を表すキー（１文字）

            dict : dictionary { "" : 1, "" : 2, ... }
        
        """
        self.df_[key] = self.df_[key].map( dict(input_dict) )   # 整数に変換

        return

    #---------------------------------------------------------
    # カテゴリデータの処理を行う関数群
    #---------------------------------------------------------
    def normalized( self, dat_X ):
        """

        """
        return

