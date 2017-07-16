# -*- coding: utf-8 -*-
import numpy

class AdaLineGD(object):
    """ADAptive Liner of classifier（分類器）by gradient descent（最急降下法）
    
    [インスタンス変数]
    pulic : publicな変数には便宜上 _ をつける
        weights_ : numpy.ndarray
            トレーニング後の重みベクトル
            [numpy.ndarray]
                最初の属性 dtype は配列の要素の型
                二番目の属性 ndim は，次元数
                三番目の属性 shape は，各次元ごとの配列の大きさをまとめたタプルで指定
        cost_ : list
            各エポックでのコスト関数（最小二乗誤差）
            Sum-of-squares cost function value in each epoch.

    private :
       lRate : float
               learning rate (0.0~1.0)
       numIter : int
               トレーニングデータの繰り返し回数のイテレータ数
    """
    def __init__(self, lRate=0.01, numIter=50):
        self.lRate = lRate
        self.numIter = numIter

    def fit(self, X_train, y_train):
        """
           ADALineを更新（学習）
           [Input]
                X_train : numpy.ndarray.shape = [numSamples, numFeatures]
                    学習データの行列

                y_train : numpy.ndarray.shape = [numSamples]
                    ラベルのベクトル

                *numpy.ndarray
                    最初の属性 dtype は配列の要素の型
                    二番目の属性 ndim は，次元数
                    三番目の属性 shape は，スカラーや，タプルによって配列の各次元の長さを表したものです． 
                    大きさが 5 のベクトルはスカラー 5 によって， 2×3 の行列はタプル (2, 3) によって表現します．
           [Output]
                self : 自身のオブジェクト
        
        """
        self.weights_ = numpy.zeros(1 + X_train.shape[1]) # numFeatures+1 個の全要素 0 の配列
        self.cost_ = []
        
        for i in range(self.numIter):
            # 活性化関数 [Activation Function] の出力の計算 Φ(w^T*x)=w^T*x
            output = self.calcNetInput(X_train)
            
            # 誤差 (y-Φ(w^T*x))の計算
            errors = (y_train - output)

            # 全ての重みの更新
            # ?  Δw=η*∑( y-Φ(w^T*x) ) (j=1,2,...,m)
            self.weights_[1:] += self.lRate * X_train.T.dot(errors) # X_train.T : X_trainの転置行列
            # w0 の更新 Δw0=η*∑( y_train-output )
            self.weights_[0] += self.lRate * errors.sum()

            # コスト関数の計算 J(w)= (1/2)*∑( y-Φ(w^T*x) )^2
            cost = (1 / 2) * (errors ** 2).sum()
            self.cost_.append(cost)

        return self


    def calcNetInput(self, X_train):
        """
        AdaLineを構成する入力層から出力層への入力を計算

        """
        numInputLayer = numpy.dot(X_train, self.weights_[1:]) + self.weights_[0]
        return numInputLayer

    def calcActivationFunction(self, X_train):
        return self.calcNetInput(X_train)

    def predict(self, X_train):
        return numpy.where(self.calcActivationFunction(X_train) > 0.0 ,  1, -1)