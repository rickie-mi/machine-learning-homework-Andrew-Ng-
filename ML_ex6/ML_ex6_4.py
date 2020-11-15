import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb   #提供了更好的可视化，相比于matplotlib
from scipy.io import loadmat
from sklearn import svm

path1 = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\spamTest.mat'
path2 = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\spamTrain.mat'
testdata = loadmat(path1)
traindata = loadmat(path2)
Xtrain = traindata['X']
ytrain = traindata['y']
Xtest = testdata['Xtest']
ytest = testdata['ytest']
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

svc = svm.SVC(C=20)
svc.fit(Xtrain,ytrain)
print('training set: {}%'.format(np.round(svc.score(Xtrain,ytrain)*100,2)))
print('testing set: {}%'.format(np.round(svc.score(Xtest,ytest)*100,2)))

kw = np.eye(1899)
spam_val = pd.DataFrame({'idx':range(1899)})
spam_val['isspam'] = svc.decision_function(kw)

print(spam_val)
#这一列数据的总体情况
print(spam_val['isspam'].describe())

decision = spam_val[spam_val['isspam']>0.05]
print(decision)

'''找出这些数字'''
path = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\iocab.txt'
voc = pd.read_csv(path, header=None, names=['idx','voc'],sep='\t')
spamvoc = voc.loc[list(decision['idx'])]
print(spamvoc)