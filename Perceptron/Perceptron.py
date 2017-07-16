# -*- coding: utf-8 -*-

import numpy

class Perceptron(object):

    """
    [クラス変数（staticな変数）]
        Nothing

    [インスタンス変数（全てpublic, private や protected でアクセスを制限することはできません]
    public : publicな変数には便宜上 _ をつける
        weigths_ : numpy.ndarray
            重みベクトル
        errors_ : numpy.ndarray
            各エポック[epoch]（トレーニング回数）での誤分類数

    private :
       lRate : float
               learning rate
       numIter : int
               トレーニングデータの繰り返し回数（イテレータ）
    
    """
    
    def __init__( self , lRate , numIter ) :
        """
        インスタンスを生成するときに自動的に呼び出される初期化用特殊メソッド
         なお、__init__() は return で値を返すとエラーになります。
         Python の場合、メソッドの第 1 引数には self と書く習慣あり

        """
        self.lRate = lRate
        self.numIter = numIter
#        self.weights_ = numpy.array()
#        self.errors_ = numpy.array()

    def fit( self, X_train, y_train ):
        """
           パーセプトロンを更新（学習）
           [Input]
                X_train : numpy.ndarray.shape = [numSamples, numFeatures]
                    学習データの行列

                y_train : numpy.ndarray.shape = [numSamples]
                    ラベルのベクトル

                *numpy.ndarray
                    最初の属性 dtype は配列の要素の型
                    二番目の属性 ndim は，次元数
                    三番目の属性 shape は，各次元ごとの配列の大きさをまとめたタプルで指定
           [Output]
                self : 自身のオブジェクト
        
        """
        # numpy.zeros() : 全要素が0の行列を生成
        self.weights_ = numpy.zeros( 1+X_train.shape[1] ) # numFeatures+1 個の配列
        self.errors_ = []

        for it in range(self.numIter):
            errors = 0

            # zip() を用いて複数シーケンス（X_train, y_train）の並列処理ループ
            for (xi, yi) in zip( X_train,y_train ):
                diffWeight = self.lRate * ( yi- self.predict(xi) )    # Δw = η*(y_i-y^_i)
                self.weights_[1:] += diffWeight*xi                    # 重みの更新
                errors += int( diffWeight != 0.0)                     # 重みの変化量が0でない（重みを変化）させる際は、誤りであったので誤りカウントをインクリメント
            
            self.errors_.append( errors )   # 反復回数[numSamples]あたりの誤りカウント量を配列の最後に挿入

        return self


    def CalcNetInput( self, X_train ):
        """
        パーセプトロンを構成する入力層から出力層への入力を計算

        
        """
        numInputLayer = numpy.dot( X_train, self.weights_[1:] ) + self.weights_[0]
        return numInputLayer

    def predict(self, X_train ):
        """
        パーセプトロンによる識別結果のクラスラベル [+1 or -1]を返す
        
        """
        # numpy.where() で条件を満たす配列のindex を取得
        # 条件 self.CalcNetInput(X_train) >= 0.0 が True なら label=1, Falseなら label=-1
        label = numpy.where( self.CalcNetInput(X_train) >= 0.0, 1,-1 )
        return label
