# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import numpy
import matplotlib.pyplot as plt

# scikit-learn ライブラリ関連
from sklearn import datasets                            # 
#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス
from sklearn.metrics import accuracy_score              # 

# 自作クラス
import Plot2D
import DecisionTree


def main():
    print("Enter main()")
    #==========================================================================================
    # 決定木 [DecisionTree] の不純度 [purity] を表す関数の作図
    # ノードの誤り率 [eror rate], 交差エントロピー関数 [cross-entropy], ジニ係数 [Gini index] 
    #==========================================================================================
    tree = DecisionTree.DecisionTree()

    #-------------------------------
    # 不純度を表す関数群の plot
    #-------------------------------
    figure = plt.figure()
    axis = plt.subplot(1,1,1)
    plt.grid(linestyle='-')
    
    tree.plotNodeErrorFunction( figure, axis )
    tree.plotCrossEntropyFunction( figure, axis )
    tree.plotGiniIndexFunction( figure, axis )

    plt.title("purity functions (i=1)")     # title
    plt.legend(loc = "upper left")          # 凡例    
    plt.tight_layout()                      # グラフ同士のラベルが重ならない程度にグラフを小さくする。

    # 図の保存＆表示
    plt.savefig("./DecisionTree_scikit-learn_1.png", dpi=300)
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
