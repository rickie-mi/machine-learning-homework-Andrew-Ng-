from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
raw_data = loadmat("D:\python_project\ML_ex8\machine-learning-ex8\ex8\ex8data1.mat")
X = raw_data['X']

def plot_row_data(X):
    '''画出原始数据的分布'''
    fig,ax = plt.subplots(figsize=(12,8))
    ax.scatter(X[:,0], X[:,1],c='b',marker='x',label='raw data')
    ax.set_xlabel("latency")
    ax.set_ylabel("throughput")
    plt.show()

plot_row_data(X)

def cal_gaussion_param(X):
    '''计算样本点的均值和方差'''
    data_mean = np.mean(X,axis=0)  #不写axis表示所有值的平均值，axis=0表示列的平均值，axis=1表示行的平均值
    data_var = np.var(X,axis=0)
    return data_mean, data_var

data_mean, data_var = cal_gaussion_param(X)
print(data_mean, data_var)

def print_contour(X,data_mean,data_var):
    '''绘制高斯分布等高线'''
    x = np.linspace(0,25,100)
    y = np.linspace(0,25,100)
    #meshgrid作用是输入x可能取值和y可能取值，得到一个矩阵保存了所有可能值
    Xval, Yval = np.meshgrid(x,y)
    Zval = np.exp( (-0.5*(Xval-data_mean[0])**2/data_var[0]) + (-0.5*(Yval-data_mean[1])**2/data_var[1]) )
    fig, ax = plt.subplots(figsize=(12,8))
    #contour用法：绘制等高线图，levels列表表示范围
    ax.contour(Xval, Yval, Zval,levels = [10**-11, 10**-7, 10**-5, 10**-3, 0.1],colors='k')
    ax.scatter(X[:,0], X[:,1],c='b',marker='x',label='raw data')
    ax.set_xlabel("latency")
    ax.set_ylabel("throughput")
    plt.show()

print_contour(X,data_mean,data_var)


p = np.zeros((X.shape[0],1))  #存放每个样本的每个特征点的概率密度函数
p[:,0] = stats.norm(data_mean[0],data_var[0]).pdf(X[:,0])*stats.norm(data_mean[1],data_var[1]).pdf(X[:,1])
#验证集数据
Xval = raw_data['Xval']
yval = raw_data['yval']
pval = np.zeros((Xval.shape[0],1))  #存放每个样本的每个特征点的概率密度函数
pval[:,0] = stats.norm(data_mean[0],data_var[0]).pdf(Xval[:,0])*stats.norm(data_mean[1],data_var[1]).pdf(Xval[:,1])



def choose_epison(pval, yval):
    '''确定合适的边界值，根据F1-score的大小'''
    best_epison=0
    best_f1=0
    step = (pval.max()-pval.min())/1000
    for epison in np.arange(pval.min(),pval.max(),step):
        pre = pval<epison
        #类似一种逻辑和
        tp = np.sum(np.logical_and(pre==1,yval==1)).astype(float)
        fp = np.sum(np.logical_and(pre==1,yval==0)).astype(float)
        fn = np.sum(np.logical_and(pre==0,yval==1)).astype(float)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)

        if f1>best_f1:
            best_f1 = f1
            best_epison = epison

    return best_f1, best_epison

best_f1, best_epison = choose_epison(pval,yval)
print(best_f1,best_epison)


'''找到不合适的点'''
outliers = np.where(p<best_epison)
print(outliers)

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1],c='b',marker='x')
ax.scatter(X[outliers[0],0], X[outliers[0],1],c='r',marker='x')
ax.set_xlabel("latency")
ax.set_ylabel("throughput")
plt.show()
