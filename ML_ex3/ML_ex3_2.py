from scipy.io import loadmat
import numpy as np
from sklearn.metrics import classification_report
path = 'D:\python_project\ML_ex3\machine-learning-ex3\ex3\ex3weights.mat'
path2 ='D:\python_project\ML_ex3\machine-learning-ex3\ex3\ex3data1.mat'
weight = loadmat(path)
data = loadmat(path2)
theta1, theta2 = weight['Theta1'], weight['Theta2']
print(theta1.shape,theta2.shape)
a1 = np.insert(data['X'], 0, values=np.ones(data['X'].shape[0]), axis=1)
y_true = data['y']
print(a1.shape)

def sigmoid_func(z):
    return 1/(1+np.exp(-z))

'''直接开始计算'''
'''第一层'''
a1 = np.matrix(a1)
theta1 = np.matrix(theta1)
a2 = sigmoid_func(a1*theta1.T)
'''第二层'''
a2 = np.insert(a2,0,values=np.ones(a2.shape[0]),axis=1)
print(a2.shape)
theta2 = np.matrix(theta2)
a3 = sigmoid_func(a2*theta2.T)
print(a3.shape)

predict_y = np.argmax(a3,axis=1)+1
print(classification_report(y_true,predict_y))
