from sklearn import svm, metrics
from voice_utils import load_csv
from voice_utils import print_result
import os

data_dir = './input_data/'

data = load_csv(os.path.join(data_dir, "train.csv"))
test = load_csv(os.path.join(data_dir, "test.csv"))

clf = svm.SVC()
clf.fit(data["features"], data["labels"])
# 训练数据集
predict = clf.predict(test["features"])
print_result(test["labels"], predict, "SVM Confusion Matrix")
