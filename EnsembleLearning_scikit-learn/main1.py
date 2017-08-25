# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import pandas
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連


# 自作クラス
import EnsembleLearningClassifier
import DataPreProcess
import Plot2D

def main():
    """
    アンサンブル学習.
    多数決方式のアンサンブル法と、単体での分類器での誤分類率.
    及び多数決方式のアンサンブル法における分類器の個数に応じた比較.
    
    """
    print("Enter main()")
    
    ensemble_clf1 = EnsembleLearningClassifier.EnsembleLearningClassifier( n_classifier = 2 )
    ensemble_clf2 = EnsembleLearningClassifier.EnsembleLearningClassifier( n_classifier = 3 )
    ensemble_clf3 = EnsembleLearningClassifier.EnsembleLearningClassifier( n_classifier = 5 )
    ensemble_clf4 = EnsembleLearningClassifier.EnsembleLearningClassifier( n_classifier = 10 )

    # データの読み込み

    #===========================================
    # 汎化性能の確認
    #===========================================
    plt.subplot(2,2,1)
    ensemble_clf1.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,2)
    ensemble_clf2.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,3)
    ensemble_clf3.plotEnsenbleErrorAndBaseError()
    plt.subplot(2,2,4)
    ensemble_clf4.plotEnsenbleErrorAndBaseError()

    plt.savefig("./EnsembleLearning_scikit-learn_1.png", dpi = 300, bbox_inches = 'tight' )
    plt.show()
    
    print("Finish main()")
    return
    
if __name__ == '__main__':
     main()