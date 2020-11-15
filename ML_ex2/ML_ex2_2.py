import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import math

path ='D:\python_project\ML_ex2\machine-learning-ex2\ex2\ex2data2.txt'
data = pd.read_csv(path,header=None,names=['test1','test2','accept'])
positive = data[data['accept'].isin(['1'])]   #不是公用一个地址
negative = data[data['accept'].isin(['0'])]

def training_show(positive,negative):
    figure,ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['test1'],positive['test2'],s=50,c='r',marker='x',label='accepted')
    ax.scatter(negative['test1'],negative['test2'],s=50,c='b',marker='o',label='unaccepted')
    ax.legend()
    ax.set_xlabel('test1')
    ax.set_ylabel('test2')
    plt.show()

data2 = data  #共用同一个地址，所以如果data2变化了，data也变化了
data2.insert(3,'ones',1)
x = data2['test1']
y = data2['test2']
degree = 6
for i in range(degree+1):
    for j in range(degree+1-i):
        if i==j and i==0:
            continue
        data2['F'+str(i)+str(j)] = np.power(x,i)*np.power(y,j)
data2.drop('test1',axis=1,inplace=True)
data2.drop('test2',axis=1,inplace=True)
print(data2.head())

def sigmoid_func(z):
    return 1/(1+np.exp(-z))

def Cost_func(theta,X,y,punishing_rate):
    '''偏差函数，-y*ln(h)-(1-y)*ln(1-h)前的系数为1/m，而punishment的系数为1/2m'''
    '''X是m*(n+1),y是m*1,theta是1*(n+1)'''
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid_func(X*theta.T)))
    second = np.multiply(1-y,np.log(1-sigmoid_func(X*theta.T)))
    ##这里一定要注意theta0不作为惩罚系数
    punish = punishing_rate/(2*len(y))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return 1/len(y)*np.sum(first-second)+punish

cols = data2.shape[1]
y = data2.iloc[:,0:1]
X = data2.iloc[:,1:cols]
theta = np.zeros(X.shape[1])
print(X.shape, y.shape, theta.shape)
#设置惩罚率为1
punishing_rate = 1
X = np.array(X)
y = np.array(y)
print(Cost_func(theta,X,y,punishing_rate))

def Gradient_func(theta,X,y,punishing_rate):
    '''计算梯度函数'''
    '''X是m*(n+1),y是m*1,theta是1*(n+1)'''
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = theta.shape[1]
    grad = np.zeros(parameters)
    minu = sigmoid_func(X*theta.T)-y
    for i in range(parameters):
        term = np.multiply(minu,X[:,i])
        if i==0:
            grad[i] = 1/len(y)*np.sum(term)
        else:
            grad[i] = 1/len(y)*np.sum(term) + punishing_rate/len(y)*theta[:,i]
            #这里采用theta[:,i]不是theta[i]的原因是theta已经是变成二维矩阵了
    return grad
print(Gradient_func(theta,X,y,punishing_rate))

result = opt.fmin_tnc(func=Cost_func,x0=theta,fprime=Gradient_func,args=(X,y,punishing_rate))
print(result)

result_theta = result[0]

def prediction(theta,X):
    '''将训练集上的数据进行预测'''
    X = np.matrix(X)
    theta = np.matrix(theta)
    predicon_y = X*theta.T
    return [1 if x>=0 else 0 for x in predicon_y]

def accuracy_prediction(theta,X,y):
    prediction_ys = prediction(theta,X)
    acc = 0
    for i in range(len(prediction_ys)):
        if (y[i]==1 and prediction_ys[i]==1) or (y[i]==0 and prediction_ys[i]==0):
            acc = acc+1
    return acc/len(prediction_ys)
print(accuracy_prediction(result_theta,X,y))

def hfunc2(x1,x2,theta):
    '''最硬核的计算六次方相加后的值'''
    sum = theta[0]
    degree = 6
    place = 0
    for i in range(degree+1):
        for j in range(degree+1-i):
            if i==j and i==0:
                continue
            sum += np.power(x1,i)*np.power(x2,j)*theta[place+1]
            place += 1
    return sum


def find_decision_boundary(theta,X,y):
    '''选取一些值，看其是否逼近0'''
    x1 = np.linspace(-1,1.5,1000)
    x2 = np.linspace(-1,1.5,1000)
    cord = [(i,j) for i in x1 for j in x2]
    x_cord,y_cord = zip(*cord)
    all_point = pd.DataFrame({'test1':x_cord,'test2':y_cord})
    all_point['hval'] =hfunc2(all_point['test1'],all_point['test2'],result_theta)

    decision = all_point[np.abs(all_point['hval'])<2*10**-3]
    #return decision.test1, decision.test2
    return decision['test1'], decision['test2']

def draw_prediction(positive,negative,theta,X,y):
    figure,ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['test1'],positive['test2'],s=50,c='r',marker='x',label='accepted')
    ax.scatter(negative['test1'],negative['test2'],s=50,c='b',marker='o',label='unaccepted')
    xx,yy = find_decision_boundary(theta,X,y)
    plt.scatter(xx, yy, c='y', s=10, label='Prediction')
    ax.legend()
    plt.show()
draw_prediction(positive,negative,theta,X,y)