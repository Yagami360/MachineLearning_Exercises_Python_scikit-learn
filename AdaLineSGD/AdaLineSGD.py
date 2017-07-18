import numpy
from numpy.random import seed

class AdaLineSGD(object):
    """
    Adaptive liner neuron of classifier by stomatical gradient decent (online learning)
    （ADALine分類器 × 確率的勾配降下法による学習 ＜オンライン学習＞）
    
    #-------------------------------------------------------------------------------------
    # インスタンス変数
    #-------------------------------------------------------------------------------------
    public : publicな変数には便宜上 _ をつける
        weights_ : numpy.ndarray
            トレーニング後の重みベクトル
            [numpy.ndarray]
                １番目の属性 dtype は配列の要素の型
                ２番目の属性 ndim は，次元数
                ３番目の属性 shape は，各次元ごとの配列の大きさをまとめたタプルで指定
        errors_ : list
            各エポックでの誤識別数

    private :
        lRate : float
           learning rate (0.0~1.0)
        numIter : int
           トレーニングデータの繰り返し回数のイテレータ数
        shuffle : bool (default: True)
            重みベクトルの循環回避用に各エポックでトレーニングデータをシャッフルするか否かの bool 値
        ramdom_state_ : int (default: None)
            シャッフルに使用する ramdom_state を設定し, 重みを初期化
        w_initialized : bool (default: False)
            重みの初期化フラグ

    """

    def __init__( self, lRate=0.01, numIter=10, shuffle=True, random_state=None ):
        """ AdaLineSGDクラスのコンストラクタ. AdaLineSGD オブジェクトを生成 """
        self.lRate = lRate
        self.numIter = numIter
        self.w_initialized = False
        self.shuffle = shuffle

        # random_state が初期値 None でない場合
        if ( random_state != None ):
            seed( random_state )        # seed() : 乱数発生の初期化, 固定した seed=random_state(int) を指定
        return

    def fit( self, X_train, y_train ):
        """
        AdaLineSGDを更新（適合）（by 確率的勾配降下法 [stomatical gradient decent]）
            [Input]
                X_train : numpy.ndarray.shape = [numSamples, numFeatures]
                    学習データの行列
                y_train : numpy.ndarray.shape = [numSamples]
                    ラベルのベクトル
                *numpy.ndarray
                    １番目の属性 dtype は, 配列の要素の型
                    ２番目の属性 ndim は，次元数
                    ３番目の属性 shape は，スカラーや，タプルによって配列の各次元の長さを表したもの 
                    大きさが 5 のベクトルはスカラー 5 によって， 2×3 の行列はタプル (2, 3) によって表現
            [Output]
                self : 自身のオブジェクト
        """
        self.initializedWeights( X_train.shape[1] ) # 重みベクトルを 0 に初期化
        self.cost_ = []                             # 平均コスト関数値のリストを初期化

        #--------------------------------------------------
        # トレーニング回数分, トレーニングデータを反復学習
        #--------------------------------------------------
        for i in range( self.numIter ):
            # シャッフル:ON の場合は, トレーニングデータをシャッフル
            if ( self.shuffle == True ):
                index = self.GetIndexShuffle( X_train, y_train )
                X_train = X_train[index]
                y_train = y_train[index]

            # 各サンプルのコスト関数の値を格納するリストを初期化
            costs = []

            #-----------------------------------------------------------
            # 各サンプルに関して、重みを更新し、コスト関数の値を計算
            #-----------------------------------------------------------
            for (xi,yi) in zip(X_train,y_train):
                costs.append( self.updateWeights(xi,yi) )

            # 全サンプルの平均コストを計算
            avgCost = sum(costs) / len(y_train)

            # 平均コストをリスト最後方に格納
            self.cost_.append( avgCost )

        return self
        
    def online_fit( self, X_train, y_train ):
        """
        AdaLineSGD を重みベクトルを再初期化することなしで更新 （オンライン学習で使用）
        
            [Input]
                X_train : numpy.ndarray.shape = [numSamples, numFeatures]
                    学習データの行列
                y_train : numpy.ndarray.shape = [numSamples]
                    ラベルのベクトル
                *numpy.ndarray
                    １番目の属性 dtype は, 配列の要素の型
                    ２番目の属性 ndim は，次元数
                    ３番目の属性 shape は，スカラーや，タプルによって配列の各次元の長さを表したもの 
                    大きさが 5 のベクトルはスカラー 5 によって， 2×3 の行列はタプル (2, 3) によって表現
            [Output]
                self : 自身のオブジェクト        
        """
        if ( self.w_initialized == False ):
            self.initializedWeights( X_train.shape[1] )
        
        # 教師データ y の要素数が 2 個以上の場合は、for ループで重みを更新
        if ( y_train.ravel().shape[0] > 1 ):
            # シャッフル:ON の場合は, トレーニングデータをシャッフル
            if ( self.shuffle == True ):
                index = self.GetIndexShuffle( X_train, y_train )
                X_train = X_train[index]
                y_train = y_train[index]
            
            #
            costs = []
            for (xi,yi) in zip(X_train,y_train):
                costs.append( self.updateWeights(xi,yi) )

            # 全サンプルの平均コストを計算
            avgCost = sum(costs) / len(y_train)
            self.cost_.append( avgCost )

        # 教師データ y の要素数が 1 個の場合（ y_train[0] のみ有効値）は、そのサンプルのみで重みを更新
        else:
            self.cost_.append( self.updateWeights(X_train, y_train) )

        return self

    def initializedWeights( self, shapes ):
        """ 重みベクトルを 0 に初期化する """
        self.weights_ = numpy.zeros( 1+shapes )
        self.w_initialized = True
        return

    def GetIndexShuffle( self, X_train, y_train ):
        """ 指定されたトレーニングデータ X_train, y_train をシャッフルしたインデックス """
        index = numpy.random.permutation( len(y_train) )  # numpy.random.permutation() : 0~99 の重複のない index の乱数を生成
        return index

    def updateWeights( self, xi, yi ):
        """ AdaLineの学習規則（勾配降下法）に従って重みを更新し、そのエポックでのコスト関数の値を返す """
        # 入力層から出力層への総入力を計算
        output = self.calcNetInput( xi )

        # 教師データとの誤差ベクトルを計算
        error = ( yi - output )

        # 重みベクトルを更新する
        self.weights_[1:] += self.lRate * xi.dot(error)
        self.weights_[0] += self.lRate * error

        # コスト関数 J の値を計算
        cost = (1/2) * (error**2)

        return cost
    
    def calcNetInput( self, X_train ):
        """ 入力層から出力層への総入力を計算する """
        return numpy.dot( X_train, self.weights_[1:] ) + self.weights_[0]
    
    def calcActivation( self, X_train ):
        """ 活性化関数を計算 """
        return self.calcNetInput( X_train )
    
    def predict( self, X_train ):
        """ 1Step 後のクラスラベルを返す"""
        return numpy.where( self.calcActivation(X_train) >= 0.0, 1, -1 )
