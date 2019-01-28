import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold, cross_val_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "(%.2f)"%(cm[i, j])
        #print t
#         plt.text(j, i, t,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Bank_confusion_matrix.png')


#Read sheet
bank = pd.read_csv('bank-additional-full.csv', sep=';',header=0)
print(bank.keys())
print(bank.shape)
#Remove Nulls
bank.dropna()
le = LabelEncoder()
#Remove duration
bank.pop('duration')
#Convert categories into numerical values
bank['job'] = le.fit_transform(bank['job'])
bank['marital'] = le.fit_transform(bank['marital'])
bank['education'] = le.fit_transform(bank['education'])
bank['default'] = le.fit_transform(bank['default'])
bank['housing'] = le.fit_transform(bank['housing'])
bank['loan'] = le.fit_transform(bank['loan'])
bank['contact'] = le.fit_transform(bank['contact'])
bank['month'] = le.fit_transform(bank['month'])
bank['day_of_week'] = le.fit_transform(bank['day_of_week'])
bank['poutcome'] = le.fit_transform(bank['poutcome'])
#Create dummies
df_dummies = pd.get_dummies(bank)
#Remove Y_NO
df_dummies.pop('y_no')
print(df_dummies.head())
#Create the histogram
fig = plt.hist(df_dummies['y_yes'])
plt.savefig('y_yes_histogram.png')
#Get X and Y
X = df_dummies.values[:,0:19]
print(X)
Y = df_dummies.values[:, 19]
print(Y)
#Split training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state=100)
#Standardization
scaler = StandardScaler()
scaler.fit(x_train)
X = scaler.transform(x_train)
X = scaler.transform(x_test)
#Classifier fitting
ann = MLPClassifier(activation='tanh', hidden_layer_sizes=(23, 23, 23), solver='lbfgs', random_state=100)
ann.fit(x_train,y_train)

#Prediction
y_pred = ann.predict(x_test)

#Confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(len(set(y_test))), normalize = False,title='Confusion matrix')

#Folds testing
ann = MLPClassifier(activation='tanh', hidden_layer_sizes=(23, 23, 23), solver='lbfgs', random_state=100)
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(X): # consider the first 40 examples
    print('Train: %s | test: %s' % (train_indices, test_indices))
    ann.fit(X[train_indices], Y[train_indices])
    print('Fold test accuracy: {} %'.format(ann.score(X[train_indices], Y[train_indices])*100))

#Feature importance
