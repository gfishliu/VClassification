from sklearn import svm, metrics

def load_csv(fname):
	labels = []
	features = []
	with open(fname, "r") as f:
		i = 0
		for line in f:
			cols = line.split(",")
			if len(cols) < 2 or i == 0: i+=1;continue
			labels.append(int(cols.pop(0)))
			vals = list(cols)
			features.append(vals)
		return {"labels": labels, "features": features}

data = load_csv("F:/GoogleDriver/Feature/train.csv")
test = load_csv("F:/GoogleDriver/Feature/test.csv")

clf = svm.SVC()
clf.fit(data["features"], data["labels"])
# 训练数据集
predict = clf.predict(test["features"])
# 预测测试集
score = metrics.accuracy_score(test["labels"], predict)
# 生成测试精度
report = metrics.classification_report(test["labels"], predict)
# 生成交叉验证的报告
print('Accuracy : %f' % (score))
# 显示数据精度
print(report)
# 显示交叉验证数据集报告
CMatrix = metrics.confusion_matrix(test["labels"], predict)
print(CMatrix)
from voice_utils import plot_confusion_matrix
plot_confusion_matrix(CMatrix, title="SVM Confusion Matrix", classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])


