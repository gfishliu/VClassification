
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd

train_data = pd.read_csv("F:/GoogleDriver/Feature/train.csv")
test_data = pd.read_csv("F:/GoogleDriver/Feature/test.csv")
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
data.drop(['labels'], axis=1, inplace=True)
train_label = train_data['labels']
test_label = test_data['labels']


#pca = PCA(n_components=100, random_state=34)
#data_pca = pca.fit_transform(data)

#Xtrain, Ytrain, xtest, ytest = train_test_split(
 #   data[0:len(train_data)], train_label, test_size=0.1, random_state=34)

clf = MLPClassifier(
    hidden_layer_sizes=(100, ),
    activation='relu',
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    random_state=34)

clf.fit(train_data.values[:, 1:], train_label)

# y_predict = clf.predict(Ytrain)
# score = metrics.accuracy_score(ytest, y_predict)
# print('Accuracy : %f' % (score))

result = clf.predict(data[len(train_data):])
score = metrics.accuracy_score(test_label, result)
print('Accuracy : %f' % (score))
print(metrics.classification_report(test_label, result))
CMatrix = metrics.confusion_matrix(test_label, result)
print(CMatrix)
from voice_utils import plot_confusion_matrix

plot_confusion_matrix(CMatrix, title="NN Confusion Matrix",
                      classes=['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'])