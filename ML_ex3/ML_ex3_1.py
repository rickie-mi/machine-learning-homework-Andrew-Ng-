import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report #这个包是评价报告
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = 'D:\python_project\ML_ex3\machine-learning-ex3\ex3\ex3data1.mat'
data = loadmat(path)
print(data)  #data是一个字典形式
'''这里导入的数据样本共5000个，其中X中是20*20pixel的图片压缩成400个特征值，而y表示训练集的结果，0用10表示，1-9用对应数字表示'''
print(data['X'].shape, data['y'].shape)

'''数据可视化，随机展示100个数据'''
sample_idx = np.random.choice(data['X'].shape[0],100)#从数据样本中选取100个编号，类似从【0，5000）选随机数
#sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
#两者效果都一样，因为choic函数第一个可以接收数字,表示终点，也可以接收一个ndarray
sample_x = data['X'][sample_idx,:]
print(sample_x.shape)

fig,ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(12,12))  #设置为True所有子图共享x轴或y轴
for i in range(10):
    for j in range(10):
        #加转置的原因是数据存放在.mat中是是转置的
        #按照黑白方式进行显示
        ax[i,j].matshow(sample_x[i*10+j,:].reshape(20,20).T, cmap=plt.cm.binary)
        plt.xticks(np.array([]))  #去除原有的坐标
        plt.yticks(np.array([]))
plt.show()

def sigmoid_func(z):
    return 1/(1+np.exp(-z))

def cost_func(theta,X,y,punishingrate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid_func(X*theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid_func(X*theta.T)))
    punish = punishingrate/(2*len(y))*np.sum(np.power(theta[:,1:theta.shape[1]],2))   #注意这里只能从1开始取!!!!
    return np.sum(first-second)/len(y) + punish

def gradient_func(theta,X,y,punishingrate):
    theta = np.matrix(theta)  #1*(n+1)
    X = np.matrix(X)   #m*(n+1)
    y = np.matrix(y)  #m*1
    grad = np.zeros(theta.shape)  #m*1,省略这行也可以，后面赋值时自动生成大小
    #利用向量化的方式，而不是写loop
    grad = 1/len(y)*((X.T*(sigmoid_func(X*theta.T)-y)).T+punishingrate*theta)   #因为theta是1*（n+1）的，所以这里第一项需要转置
    grad[0,0] = 1/len(y)*np.sum(np.multiply(sigmoid_func(X*theta.T)-y,X[:,0]))
    return np.array(grad).ravel()

def one_vs_all(X,y,num_labels,punishingrate):
    rows = X.shape[0]
    params = X.shape[1]
    print(params)
    '''all_theta是定义为分类类别数*特征数量的二维array'''
    all_theta = np.zeros((num_labels,params+1))  #这里+1的目的是由于theta0还没有加进去
    X = np.insert(X,0,values=np.ones(rows),axis=1)
    for i in range(num_labels):
        solo_theta = np.zeros(params+1)
        solo_y = [1 if yy==i+1 else 0 for yy in y]  #这里+1是因为0用10来表示，所以依次+1
        solo_y = np.array(solo_y).reshape(rows,1)
        #solo_y = np.reshape(solo_y, (rows,1))
        '''对每一个类别进行学习'''
        #print(len(X),len(solo_y))
        solo_result = minimize(fun=cost_func,x0=solo_theta,args=(X,solo_y,punishingrate),method='TNC',jac=gradient_func)
        all_theta[i,:] = solo_result.x
    return all_theta

np.unique(data['y'])
all_theta = one_vs_all(data['X'],data['y'],10,1)
print(all_theta)

def predict_all(all_theta,X):
    all_theta = np.matrix(all_theta)
    X = np.matrix(X)
    X = np.insert(X, 0, values=np.ones(X.shape[0]),axis=1)
    predict_y = sigmoid_func(X*all_theta.T)
    argmax = np.argmax(predict_y,axis=1)  #1是按照行比较，0是按照列比较
    argmax = argmax+1   #因为是1-10
    return argmax

y_predict = predict_all(all_theta, data['X'])
print(classification_report(data['y'],y_predict))
'''https://blog.csdn.net/akadiao/article/details/78788864'''