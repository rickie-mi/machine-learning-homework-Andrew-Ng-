import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''输出一个5*5单位矩阵'''
A = np.eye(5)
print(A)

'''定义J(theta)'''
def computeCost(X,y,theta):
    inner = np.power((X*theta.T-y),2)
    return sum(inner)/(2*len(y))

'''获取数据并图像可视化'''
path = 'D:\python_project\ML_ex1\machine-learning-ex1\ex1\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['population','profits'])
print(data.head())   #默认输出前五组数据
#data.plot(kind='scatter',x='population', y='profits',figsize=(12,8))
#plt.scatter(data['population'],data['profits'],linewidths=1)
#plt.show()

data.insert(0,'ones',1)  #加一行全部为1的列
#获取需要学习的X和y
loc = data.shape[1]
X = data.iloc[:,:loc-1]    #iloc是一种截取
y = data.iloc[:,loc-1:loc]
#print(X.head())
#print(y.head())
#print(data.head())
'''将数据转化成矩阵形式'''
print(y.shape)
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(y.shape)
print(computeCost(X,y,theta))

def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = X*theta.T-y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(y))*np.sum(term))
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta,cost

alpha = 0.01
iters =1500
g,cost = gradientDescent(X,y,theta,alpha,iters)
print(g)

'''预测35000和70000小吃摊利润'''
prediction1 = [1,3.5]*g.T
prediction2 = [1,7]*g.T
print(prediction1, prediction2)

'''划出拟合曲线'''
x = np.linspace(data.population.min(),data.population.max(),100)
f = g[0,0]+g[0,1]*x
'''
plt.scatter(data['population'],data['profits'],linewidths=1)
plt.plot(x,f,'r')
plt.xlabel('Population')
plt.ylabel('profits')
plt.title('Predicted Profit vs. Population Size')
plt.show()
'''
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.profits, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


iter = np.array(range(1500))
cost.flatten()
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(iter,cost,label='cost function')
ax.set_title('cost function with iteration')
ax.set_xlabel('iteration')
ax.set_ylabel('cost function')
#plt.show()

path2='D:\python_project\ML_ex1\machine-learning-ex1\ex1\ex1data2.txt'
data2 = pd.read_csv(path2,header=None,names=['size','bedrooms','price'])
##特征归一化
data2 = (data2-data2.mean())/data2.std()
#print(data2.head())
loc2 = data2.shape[1]
X2 = data2.iloc[:,0:loc2-1]
y2 = data2.iloc[:,loc2-1:loc2]
X2.insert(0,'ones',1)
theta2 = np.matrix([0,0,0])
X2 = np.matrix(X2)
y2 = np.matrix(y2)
print(computeCost(X2,y2,theta2))
g2, cost2 = gradientDescent(X2,y2,theta2,alpha,iters)
print(g2)

def normalEqu(X,y):
    A = X.T.dot(X)    #表示矩阵相乘，也可以np.dot(X,X.T)
    return np.linalg.inv(A).dot(X.T).dot(y)
theta2 = normalEqu(X2,y2)
print(theta2)