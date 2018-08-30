from sklearn.ensemble import RandomForestClassifier
from voice_utils import print_result
import pandas as pd
import time
import os

# 数据路径
data_dir = './input_data/'
out_dir = './output_data'


# 加载数据
def opencsv():
	# 使用 pandas 打开
	train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
	test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
	data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
	data.drop(['labels'], axis=1, inplace=True)  # 去除labels 列

	return train_data, test_data, data


# 训练模型
def trainModel(X_train, y_train):
	print('Train RF...')
	clf = RandomForestClassifier(
		n_estimators=10,
		max_depth=10,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=34)
	clf.fit(X_train, y_train)  # 训练rf
	return clf


# 存储模型
def storeModel(model, filename):
	import pickle
	with open(filename, 'wb') as fw:
		pickle.dump(model, fw)


# 加载模型
def getModel(filename):
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)


# 结果输出保存
def saveResult(result, csvName):
	i = 0
	n = len(result)
	print('the size of test set is {}'.format(n))
	with open(os.path.join(out_dir, 'Result_sklearn_RF.csv'), 'w') as fw:
		fw.write('{},{}\n'.format('VoiceId', 'Label'))
		for i in range(1, n + 1):
			fw.write('{},{}\n'.format(i, result[i - 1]))
	print('Result saved successfully... and the path = {}'.format(csvName))


def trainRF():
	start_time = time.time()
	# 加载数据
	train_data, test_data, feature = opencsv()
	train_label = train_data['labels']
	train_feature = train_data.values[0:, 1:]  # [行， 列]
	print("load data finish")
	stop_time_l = time.time()
	print('load data time used:%f s' % (stop_time_l - start_time))

	startTime = time.time()
	rfClf = trainModel(train_feature, train_label)

	# 保存结果
	storeModel(feature[len(train_data):], os.path.join(out_dir, 'Result_sklearn_rf.pcaPreData'))
	storeModel(rfClf, os.path.join(out_dir, 'Result_sklearn_rf.model'))
	print("finish!")
	stopTime = time.time()
	print('TrainModel store time used:%f s' % (stopTime - startTime))


def preRF():
	startTime = time.time()
	# 加载模型和数据
	clf = getModel(os.path.join(out_dir, 'Result_sklearn_rf.model'))
	pcaPreData = getModel(os.path.join(out_dir, 'Result_sklearn_rf.pcaPreData'))

	# 结果预测
	predict_result = clf.predict(pcaPreData)

	# 结果的输出
	saveResult(predict_result, os.path.join(out_dir, 'Result_sklearn_rf.csv'))
	test_labels = pd.read_csv(os.path.join(data_dir, 'test.csv'))['labels']
	print_result(test_labels, predict_result, "RF Confusion Matrix")

	print("finish!")
	stopTime = time.time()
	print('PreModel load time used:%f s' % (stopTime - startTime))



if __name__ == '__main__':
	# 训练并保存模型
	trainRF()
	# 加载预测数据集
	preRF()
