import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from voice_utils import print_result


data_dir = './input_data/'


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    data1 = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    train_data = data.values[0:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[0:, 0]  # 读取列表的第一列
    test_data = data1.values[0:, 1:]  # 测试全部测试个数据
    test_lable = data1.values[0:, 0]
    return train_data, train_label, test_data, test_lable


def saveResult(result, csvName):
    with open(csvName, 'w') as myFile:  # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
        # python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])
    print('Saved successfully...')  # 保存预测结果


def knnClassify(trainData, trainLabel, k_value):
    print('k value: %d' % (k_value))
    knnClf = KNeighborsClassifier(k_value)  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, np.ravel(trainLabel))  # ravel Return a contiguous flattened array.
    return knnClf

def dRecognition_knn():
    start_time = time.time()

    # 加载数据
    trainData, trainLabel, testData, TestLabel = opencsv()
    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))


    for i in range(1, 10):
        # 模型训练
        knnClf = knnClassify(trainData, trainLabel, i)

        # 结果预测
        testLabel = knnClf.predict(testData)

        score = metrics.accuracy_score(TestLabel, testLabel)
        print('Accuracy : %f' % (score))

    # 结果的输出
    print("finish!")
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))

    #k=1时，报告分析↓
    knnClf = knnClassify(trainData, trainLabel, 1)
    testLabel = knnClf.predict(testData)
    print_result(TestLabel, testLabel, "KNN Confusion Matrix")

if __name__ == '__main__':
    dRecognition_knn()
