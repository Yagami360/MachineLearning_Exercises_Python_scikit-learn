# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn.svm import SVC                             # 
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.pipeline import Pipeline

# 自作クラス
import Plot2D
import DataPreProcess


def main():
    """
    機械学習パイプラインによる、機械学習処理フロー（scikit-learn ライブラリの Pipeline クラスを使用）
    混同行列とROC曲線によるモデルの汎化能力の評価
    """
    print("Enter main()")
    
    # データの読み込み
    prePro = DataPreProcess.DataPreProcess()
    prePro.setDataFrameFromCsvFile(
        "https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wdbc/wdbc.data"
    )
    #prePro.print( "Breast Cancer Wisconsin dataset" )
    
    dat_X = prePro.df_.loc[:, 2:].values
    dat_y = prePro.df_.loc[:, 1].values

    #===========================================
    # 前処理 [PreProcessing]
    #===========================================
    # 欠損データへの対応
    #prePro.meanImputationNaN()

    # ラベルデータをエンコード
    prePro.encodeClassLabelByLabelEncoder( colum = 1 )
    prePro.print( "Breast Cancer Wisconsin dataset" )

    # データをトレードオフデータとテストデータに分割
    X_train, X_test, y_train, y_test \
    = DataPreProcess.DataPreProcess.dataTrainTestSplit( X_input = dat_X, y_input = dat_y, ratio_test = 0.2 )

    #-------------------------------------------
    # Pipeline の設定
    #-------------------------------------------
    # パイプラインに各変換器、推定器を設定
    pipe_csvm = Pipeline(
                      steps = [                                     # タプル (任意の識別文字, 変換器 or 推定器のクラス) で指定
                                  ( "scl", StandardScaler() ),      # 正規化 : 変換器のクラス（fit() 関数を持つ）
                                  ('clf', SVC( C=10.0, kernel="rbf", gamma = 0.01 ) )   # C-SVM : 推定器のクラス（predict()関数を持つ）
                              ]
                  )

    
    # パイプラインに設定した変換器の fit() 関数を実行
    pipe_csvm.fit( X_train, y_train )

    # 予想値
    y_predict = pipe_csvm.predict( X_test )
    
    # pipeline オブジェクトの内容確認
    print( "Pipeline.get_params( deep = True ) : \n", pipe_csvm.get_params( deep = True ) )
    print( "Pipeline.get_params( deep = False ) : \n", pipe_csvm.get_params( deep = False ) )

    print( "Pipeline.predict( X_test ) : \n", y_predict )
    print( "Test Accuracy: %.3f" % pipe_csvm.score( X_test, y_test ) )

    
    #-------------------------------------------
    # 混同行列 [confusion Matrix]
    #-------------------------------------------
    # テストデータと予想データから混同行列を作成
    mat_confusion = confusion_matrix( y_true = y_test, y_pred = y_predict )
    print( "mat_confusion : \n", mat_confusion )

    # 混同行列のヒートマップを作図
    Plot2D.Plot2D.drawHeatMapFromConfusionMatrix( mat_confusion = mat_confusion )
    plt.title("Heat Map of Confusion Matrix \n classifiler : RBF-kernel SVM (C = 10.0, gamma = 0.01)")

    plt.savefig("./MachineLearningPipeline_scikit-learn_4.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()

    #-------------------------------------------
    # 適合率、再現率、F1 スコア
    #-------------------------------------------
    #print( 'Precision: %.3f' % precision_score(y_true = y_test, y_pred = y_predict) )
    #print( 'Recall: %.3f' % recall_score(y_true = y_test, y_pred = y_predict) )
    #print( 'F1: %.3f' % f1_score(y_true=y_test, y_pred=y_predict) )


    #-------------------------------------------
    # ROC 曲線
    #-------------------------------------------
    
    
    
    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()