import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import metrics

features = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']


def plot_confusion_matrix(cm, classes=features,
						  normalize=False,
						  title='Confusion matrix',
						  cmap="Blues"):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	# 是否进行标准化
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


def load_csv(fname):
	labels = []
	features = []
	with open(fname, "r") as f:
		i = 0
		for line in f:
			cols = line.split(",")
			if len(cols) < 2 or i == 0: i += 1; continue
			labels.append(int(cols.pop(0)))
			vals = list(cols)
			features.append(vals)
	return {"labels": labels, "features": features}


# 打印结果测试精度，交叉验证报告，混淆矩阵
def print_result(y_test, y_predict, title):
	# 生成测试精度
	score = metrics.accuracy_score(y_test, y_predict)
	# 生成交叉验证的报告
	report = metrics.classification_report(y_test, y_predict)
	# 生成混淆矩阵
	CMatrix = metrics.confusion_matrix(y_test, y_predict)

	print('Accuracy : %f' % (score))
	# 显示数据精度
	print('Report : \n', report)
	# 显示交叉验证数据集报告
	print('Confusion Matrix : \n', CMatrix)
	plot_confusion_matrix(CMatrix, title=title)
