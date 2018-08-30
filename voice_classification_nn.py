from sklearn.neural_network import MLPClassifier
from voice_utils import print_result
import pandas as pd
import os

data_dir = './input_data/'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['labels'], axis=1, inplace=True)
train_label = train_data['labels']
test_label = test_data['labels']

clf = MLPClassifier(
	hidden_layer_sizes=(100,),
	activation='relu',
	alpha=0.0001,
	learning_rate='constant',
	learning_rate_init=0.001,
	max_iter=200,
	shuffle=True,
	random_state=34)

clf.fit(train_data.values[:, 1:], train_label)

result = clf.predict(data[len(train_data):])
print_result(test_label, result,  "NN Confusion Matrix")
