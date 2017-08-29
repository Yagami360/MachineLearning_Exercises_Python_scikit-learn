# -*- coding: utf-8 -*-

"""
    更新情報
    [17/08/18] : 学習曲線を標準偏差のバラツキで塗りつぶして描写する関数 drawLearningCurve() を追加
               : 検証曲線を標準偏差のバラツキで塗りつぶして描写する関数 drawValidationCurve() を追加
    [17/08/18] : ヒートマップの描写関数 drawHeapMap() を追加
    [17/08/21] : ヒートマップの描写関数を改名＆修正（drawHeatMapFromGridSearch()）
    [17/08/22] : ROC曲線の描写関数 `drawROCCurveFromTrainTestIterator()` 追加
    [17/08/27] : ROC曲線の描写関数 `drawROCCurveFromClassifiers()` 追加

"""

import numpy
import pandas

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve                   # ROC曲線
from sklearn.metrics import auc                         # AUC
from scipy import interp                                # （AUCを計算するための）補間処理


class Plot2D(object):
    """
    ２次元の図を描写をサポートする関数群からなるクラス
    """
    
    def __init__( self ):
        self.mainTitle = "mainTitle"
    
    @ staticmethod
    def drawDiscriminantRegions( dat_X, dat_y, classifier, list_test_idx = None, resolusion = 0.02 ):
        """ 識別器 [classifier] による識別領域を色分けで描写する """
        
        # 識別クラス数に対応したMAPの作成（最大５クラス対応）
        tuple_makers = ( "s","x","+","^","v" )                          # タプル（定数リスト）
        tuple_colors = ( "red","blue","lightgreen", "gray", "cyan" )    # 塗りつぶす色を表すタプル（定数リスト）
        numClass = len( numpy.unique(dat_y) )                           # numpy.unique() : 指定したarray変数の要素の重複をなくしたものを返す,更にlen() でユニークな値の数取得
        cmap = ListedColormap( tuple_colors[0:numClass] )               # plt.scatter() の引数で使用

        # plot the decision surface
        x1_min = dat_X[:, 0].min() - 1
        x1_max = dat_X[:, 0].max() + 1
        x2_min = dat_X[:, 1].min() - 1
        x2_max = dat_X[:, 1].max() + 1

        meshgrids = numpy.meshgrid(                                     # マス目を作る ( 戻り値:numpy.ndarray )
                        numpy.arange( x1_min, x1_max, resolusion ),     # numpy.arang(): min~max by resolution
                        numpy.arange( x2_min, x2_max, resolusion )
                    )
        # 入力データ datX のx1軸、x2軸の値の全ての組み合わせ
        xx1 = meshgrids[0]
        xx2 = meshgrids[1]

        #
        classifier.fit( dat_X, dat_y)

        # ? 値の全ての組み合わせを１次元配列に変換 numpy.array( [xx1.ravel(), xx2.ravel()] ) し、
        # classifierに設定されている predict（予想）を実行
        Z = classifier.predict( 
                numpy.array( [xx1.ravel(), xx2.ravel()] ).T
            )
        # ? 予測結果を元のグリッドポイントサイズに変換
        Z = Z.reshape( xx1.shape )  # numpy.ndarray の属性 shape は，各次元ごとの配列の大きさをまとめたタプルで指定

        # 等高線plotで識別領域を塗りつぶす
        plt.contourf( xx1, xx2, Z, alpha=0.4, cmap=cmap )

        # 図の軸の範囲指定
        plt.xlim( xx1.min(), xx1.max() )
        plt.ylim( xx2.min(), xx2.max() )

        # 識別クラス毎に、入力データ dat_X, dat_y の散布図 plot
        for (idx, cl) in enumerate( numpy.unique(dat_y) ): # enumerate():idx と共に clもloop
            plt.scatter(
                x = dat_X[dat_y == cl, 0], 
                y = dat_X[dat_y == cl, 1],
                alpha = 0.8, 
                c = cmap(idx),
                edgecolor = 'black',
                marker = tuple_makers[idx],
                label = cl
            )

        # テスト用サンプルデータを強調表示
        if (list_test_idx != None):
            X_test = dat_X[list_test_idx, :]
            y_test = dat_y[list_test_idx]
            plt.scatter(
                X_test[:, 0], X_test[:, 1],
                c='',
                alpha=1.0,
                edgecolor='black',
                linewidths=1,
                marker='o',
                s=55, 
                label='test set'
            )

        # グラフ同士のラベルが重ならない程度にグラフを小さくする。
        plt.tight_layout()

        return

    @ staticmethod
    def drawLearningCurve( 
        train_sizes, 
        train_means, train_stds, 
        test_means, test_stds, 
        train_label = "training accuracy", test_label = "k-fold cross validation accuracy",
        input_alpha = 0.15 ):
        """
        学習曲線を平均値±標準偏差の幅で塗りつぶて描写する.

        [Input]
            train_sizes : numpy 1 次元配列
                トレーニングデータの分散値のリスト（横軸の値：トレーニングサンプル数に対応）
            train_means : numpy 1 次元配列
                トレーニングデータの平均値のリスト（plot点に対応）
            train_stds : numpy 1 次元配列
                トレーニングデータの標準偏差のリスト（y軸の±方向の塗りつぶしに対応）

            test_sizes : numpy 1 次元配列
                テストデータの分割値のリスト（横軸の値：トレーニングサンプル数に対応）
            test_means : numpy 1 次元配列
                テストデータの平均値のリスト（plot点に対応）
            test_stds : numpy 1 次元配列
                テストデータの標準偏差のリスト（y軸の±方向の塗りつぶしに対応）
            
            input_alpha : float
                塗りつぶすの透明度
        """
        #-----------------------------------------------
        # トレーニングデータ
        #-----------------------------------------------
        # 平均値を plot （青色の●）
        plt.plot(
            train_sizes, train_means,
            color = 'blue', 
            marker = 'o',
            markersize = 5, 
            label = train_label
        )
        
        # fill_between() 関数で平均値±標準偏差の幅を塗りつぶす
        plt.fill_between(
            train_sizes,
            train_means + train_stds,    # + 方向（上方向）
            train_means - train_stds,    # - 方向（下方向）
            alpha = input_alpha, 
            color = 'blue'
        )

        #-----------------------------------------------
        # テストデータ
        #-----------------------------------------------
        # 平均値を plot
        plt.plot(
            train_sizes, test_means,
            color = 'green', 
            linestyle = '--',
            marker='s', 
            markersize = 5,
            label = test_label
        )

        # fill_between() 関数で平均値±標準偏差の幅を塗りつぶす
        plt.fill_between(
            train_sizes,
            test_means + test_stds,
            test_means - test_stds,
            alpha = input_alpha, 
            color = 'green'
        )

        plt.grid()
        
        return


    @ staticmethod
    def drawValidationCurve( 
        param_range, 
        train_means, train_stds, 
        test_means, test_stds, 
        train_label = "training accuracy", test_label = "k-fold cross validation accuracy",
        input_alpha = 0.15 ):
        """
        学習曲線を平均値±標準偏差の幅で塗りつぶて描写する.

        [Input]
            param_range : numpy 1 次元配列
                モデルのパラメータのリスト（横軸の値：パラメータ値に対応）
            train_means : numpy 1 次元配列
                トレーニングデータの平均値のリスト（plot点に対応）
            train_stds : numpy 1 次元配列
                トレーニングデータの標準偏差のリスト（y軸の±方向の塗りつぶしに対応）

            test_sizes : numpy 1 次元配列
                テストデータの分散値のリスト（横軸の値：トレーニングサンプル数に対応）
            test_means : numpy 1 次元配列
                テストデータの平均値のリスト（plot点に対応）
            test_stds : numpy 1 次元配列
                テストデータの標準偏差のリスト（y軸の±方向の塗りつぶしに対応）
            
            input_alpha : float
                塗りつぶすの透明度
        """
        #-----------------------------------------------
        # トレーニングデータ
        #-----------------------------------------------
        # 平均値を plot （青色の●）
        plt.plot(
            param_range, train_means,
            color = 'blue', 
            marker = 'o',
            markersize = 5, 
            label = train_label
        )
        
        # fill_between() 関数で平均値±標準偏差の幅を塗りつぶす
        plt.fill_between(
            param_range,
            train_means + train_stds,    # + 方向（上方向）
            train_means - train_stds,    # - 方向（下方向）
            alpha = input_alpha, 
            color = 'blue'
        )

        #-----------------------------------------------
        # テストデータ
        #-----------------------------------------------
        # 平均値を plot
        plt.plot(
            param_range, test_means,
            color = 'green', 
            linestyle = '--',
            marker='s', 
            markersize = 5,
            label = test_label
        )

        # fill_between() 関数で平均値±標準偏差の幅を塗りつぶす
        plt.fill_between(
            param_range,
            test_means + test_stds,
            test_means - test_stds,
            alpha = input_alpha, 
            color = 'green'
        )

        plt.grid()
        
        return

    @ staticmethod
    def drawHeatMapFromGridSearch( dat_Z, dat_x, dat_y, input_cmap = "Blues" ):
        """
        グリッドサーチのヒートマップを作図する.

        [Input]
            dat_Z : 2 次元 list
                ヒートマップの各グリッドの値（Matrix）
            dat_x : 1 次元 list
                ヒートマップの x 軸の目盛りのリスト
            dat_y : 1 次元list
                ヒートマップの y 軸の目盛りのリスト
            input_cmap : Colour_map
                ヒートマップのカラーマップ
        """
        df_heapMap = dat_Z

        # ヒートマップを作図する
        seaborn.heatmap(
            data = df_heapMap,          # ndarray 形式に変換可能な 2 次元のデータセット指定
            vmin = 0.0, vmax = 1.0,     # カラーマップと値の範囲を関連付ける必要がある際に最小値、最大値を指定し
            cmap = input_cmap,          # Colour_map
            center = 0.5,               # olormap の 中心とする値。(デフォルト値: None)
            annot = True,               # True に設定すると、セルに値を出力
            fmt = '.3g',                # 数値の桁の調整
            xticklabels = dat_x,        # x 軸目盛り
            yticklabels = dat_y         # y 軸目盛り
        )

        return

    @ staticmethod
    def drawHeatMapFromConfusionMatrix( mat_confusion, input_vmin = 0, input_vmax = 100, input_cmap = "Blues" ):
        """
        混同行列のヒートマップを作図する.

        [Input]
            mat_confusion : 
                混同行列

        """
        
        """
        # 文字列に変換し, 文字を付加
        mat_confusion[0,0].append("TP [true positive]")
        mat_confusion[0,1].append("FN [false negative]")
        mat_confusion[1,0].append("FP [false positive]")
        mat_confusion[1,1].append("TN [true negative]")
        """

        # ヒートマップを作図する
        seaborn.heatmap(
            data = mat_confusion,       # ndarray 形式に変換可能な 2 次元のデータセット指定
            vmin = input_vmin,          # カラーマップと値の範囲を関連付ける必要がある際の最小値
            vmax = input_vmax,          # カラーマップと値の範囲を関連付ける必要がある際の最大値
            cmap = input_cmap,          # Colour_map
            center = None,              # olormap の 中心とする値。(デフォルト値: None)
            annot = True,               # True に設定すると、セルに値を出力
            fmt = 'd',                  # テキストで出力
            xticklabels = ["P","N"],    # x 軸目盛り
            yticklabels = ["P","N"]     # y 軸目盛り
        )
        
        plt.title( "heat map of confusion matrix" )
        plt.xlabel( "predicted label" )
        plt.ylabel( "true label" )

        return

    @ staticmethod
    def drawROCCurveFromTrainTestIterator( classifiler, iterator, X_train, y_train, X_test, y_test, positiveLabel = 1 ):
        """
        トレーニングデータとテストデータを分割するイテレータから、ROC曲線を描写する.

        [Input]
            classifiler : 推定器クラスのオブジェクト
                fit() 関数と predict() 関数が実装されたクラスのオブジェクト
            iterator : list
                イテレータ
        [Output]
            figure : matplotlib.figure クラスのオブジェクト
                描画される部品を納めるコンテナクラス ( Artist の派生クラス )

        """
        # Figure クラスのオブジェクト作成＆グラフサイズを設定
        figure = plt.figure( figsize = (7, 5) )

        # ROC 曲線を構成する偽陽性率 [FPR] と真陽性率 [TPR] の初期化
        means_tpr = 0.0                         # 
        means_fpr = numpy.linspace(0, 1, 100)   # [0,1] の範囲（確率）を 100 個で分割
        #all_tpr   = []                          # 空のリストで初期化 

        #---------------------------------------------------------------------------------------
        # iterator 内の分割された ( train, test ) のペアでループ処理 (enumerate で並列ループ)
        # イテレータ毎に ROC曲線 & AUC の描写処理
        #---------------------------------------------------------------------------------------
        for it, (train, test) in enumerate( iterator ):
            #print("X_train[train] : \n", X_train[train] )
            #print("y_train[train] : \n", y_train[train] )

            # トレーニングデータで推定器 classifiler を学習 fit()
            predict = classifiler.fit( X_train[train], y_train[train] )
            #print("predict : \n", predict )

            # test データの予想所属確率を predict_proba() で算出
            proba = predict.predict_proba( X_train[test] )
            #print("predict_proba : \n", proba )

            # 実際の所属確率と予想の所属確率から roc_curve() 関数で ROC 曲線の性能値（FPR,TPR）を計算
            fpr, tpr, thresholds = roc_curve( 
                                       y_true = y_train[test],      # True binary labels in range {0, 1} or {-1, 1} 
                                       y_score = proba[:, 1],       # Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions 
                                       pos_label = positiveLabel    # positive と見なすラベルの値
                                   )
            #print("roc_curve() retrun FPR : \n", fpr )
            #print("roc_curve() retrun TPR : \n", tpr )
            #print("roc_curve() retrun thresholds : \n", thresholds )

            # 得られた fpr (x軸値) と tpr (y軸値) の線形補間処理
            means_tpr += interp( means_fpr, fpr, tpr ) 
            
            #print("means_tpr : \n", means_tpr )
            means_tpr[0] = 0.0          # ?
            #print("means_tpr : \n", means_tpr )

            # AUC 値を計算
            roc_auc = auc( fpr, tpr )
            #print("roc_auc : \n", roc_auc )

            # 計算したROC 曲線の性能を plot
            plt.plot(
                fpr, tpr,   # 偽陽性率 [FPR] と真陽性率 [TPR]
                lw = 1,
                label = 'ROC k=%d fold CV (AUC = %0.2f)' % ( it+1, roc_auc )
            )

        # ROC の平均を plot
        means_tpr /= len(iterator)
        means_tpr[-1] = 1.0
        mean_auc = auc( means_fpr, means_tpr )
        #print("means_tpr : \n", means_tpr )
        
        plt.plot(
            means_fpr, means_tpr, 
            'k--',
            label = 'mean ROC (AUC = %0.2f)' % mean_auc, 
            lw = 2
        )

        # perfect performance 時の ROC 曲線 plot
        plt.plot(
            [0, 0, 1], [0, 1, 1],
            lw=2,
            linestyle=':',
            color='black',
            label='perfect performance (AUC = 1.00)'
        )

        # 当て推量時の ROC 曲線 & AUC値 plot
        plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            color = (0.6, 0.6, 0.6),
            label='random guessing (AUC =0.50)'
        )

        #
        plt.title( "ROC Curve [Receiver Operator Characteristic Curve]" )
        plt.xlabel( "FPR : false positive rate" )
        plt.ylabel( "TPR : true positive rate" )
        
        plt.xlim( [-0.05, 1.05] )
        plt.ylim( [-0.05, 1.05] )
        plt.legend( loc = 'best' )

        #plt.grid()
        #plt.tight_layout()

        return figure


    @staticmethod
    def drawROCCurveFromClassifiers(  classifilers, class_labels, X_train, y_train, X_test, y_test, positiveLabel = 1  ):
        """
        トレーニングデータとテストデータを分割するイテレータから、ROC曲線を描写する.

        [Input]
            classifilers : 推定器クラスのオブジェクト
                fit() 関数と predict() 関数が実装されたクラスのオブジェクト
            
            class_labels : list <str>

        """
        # 分類器 classifers に対応したMAPの作成（最大５クラス対応）
        #tuple_makers = ( "s","x","+","^","v" )                          # タプル（定数リスト）
        #tuple_colors = ( "red","blue","lightgreen", "gray", "cyan" )    # 塗りつぶす色を表すタプル（定数リスト）
        #tuple_linestyle = ( 'k--', '-', '-.', '--', "---" )
        
        # classifilers 内の各弱識別器 clf の ROC 曲線を作図
        for ( clf, label ) in zip( classifilers, class_labels ):
            # トレーニングデータで推定器 classifiler を学習 fit()
            predict = clf.fit( X_train, y_train )
            #print("predict : \n", predict )

            # test データの予想所属確率を predict_proba() で算出
            proba = predict.predict_proba( X_test )
            #print("predict_proba : \n", proba )

            # 実際の所属確率と予想の所属確率から roc_curve() 関数で ROC 曲線の性能値（FPR,TPR）を計算
            fpr, tpr, thresholds = roc_curve( 
                                       y_true = y_test,            # True binary labels in range {0, 1} or {-1, 1} 
                                       y_score = proba[:, 1],       # Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions 
                                       pos_label = positiveLabel    # positive と見なすラベルの値
                                   )
            
            # AUC 値を計算
            roc_auc = auc( fpr, tpr )
            #print("roc_auc : \n", roc_auc )

            # 計算したROC 曲線の性能を plot
            plt.plot(
                fpr, tpr,   # 偽陽性率 [FPR] と真陽性率 [TPR]
                lw = 2,
                label = '%s (AUC = %0.2f)' % (label, roc_auc)
            )

        # perfect performance 時の ROC 曲線 plot
        plt.plot(
            [0, 0, 1], [0, 1, 1],
            lw=1,
            linestyle=':',
            color='black',
            label='perfect performance (AUC = 1.00)'
        )

        # 当て推量時の ROC 曲線 & AUC値 plot
        plt.plot(
            [0, 1], [0, 1],
            lw=1,
            linestyle='--',
            color = (0.6, 0.6, 0.6),
            label='random guessing (AUC =0.50)'
        )

        #
        plt.title( "ROC Curve [Receiver Operator Characteristic Curve]" )
        plt.xlabel( "FPR : false positive rate" )
        plt.ylabel( "TPR : true positive rate" )
        
        plt.xlim( [-0.05, 1.05] )
        plt.ylim( [-0.05, 1.05] )
        plt.legend( loc = 'best' )

        return