import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb   #提供了更好的可视化，相比于matplotlib
from scipy.io import loadmat
from sklearn import svm

path = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\ex6data1.mat'
raw_data = loadmat(path)
data = pd.DataFrame(raw_data['X'], columns=['X1','X2'])  #列的名称
data['y'] = raw_data['y']
#print(data.head())
'''
画出原来的这些点所在的位置，正负样本点分开标记
'''
def plot_init_data(data,fig,ax):
    positive = data[data['y'].isin(['1'])]
    negative = data[data['y'].isin(['0'])]
    ax.scatter(positive['X1'],positive['X2'],marker='x',c='r',label='Positive')
    ax.scatter(negative['X1'],negative['X2'],marker='o',c='b',label='Negative')
'''
fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data,fig,ax)
ax.legend()
plt.show()
'''

'''
Most SVM software packages automatically add the extra feature x0 = 1 for you and automatically 
take care of learning the intercept term theta0. So when passing
your training data to the SVM software, there is no need to add this extra feature x0 = 1 yourself. 
'''
#print(data['X1','X2'])   这句话是错的，必须得写成data[['X1','X2']] ！！！
svc = svm.LinearSVC(C=1,loss='hinge',max_iter=1000)
svc.fit(data[['X1','X2']], data['y'])
print(svc.score(data[['X1','X2']],data['y']))

def find_decision_bundary(svc, x1min, x1max, x2min, x2max, differ):
    x1 = np.linspace(x1min,x1max,1000)
    x2 = np.linspace(x2min,x2max,1000)
    '''这里不能这样写，否则就只有1000个点，实际上它是需要这个二维区间所有的点，也就是1000*1000个点
    c_val = pd.DataFrame({'x1':x1, 'x2':x2})
    print(c_val)
    '''
    cordinates = [(x,y) for x in x1 for y in x2]
    cor_x1, cor_x2 = zip(*cordinates)
    c_val = pd.DataFrame({'x1': cor_x1, 'x2': cor_x2})
    print(c_val)
    c_val['c_val'] = svc.decision_function(c_val[['x1', 'x2']])
    decission = c_val[np.abs(c_val['c_val'])<differ]
    return decission['x1'], decission['x2']

x1, x2 = find_decision_bundary(svc, 0, 4, 1.5, 5, 2 * 10**-3)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x1,x2,s=10,label='prediction')
plot_init_data(data,fig,ax)
ax.set_title('SVM (C=1) Decision Boundary')
ax.legend()
plt.show()

'''
在惩罚系数为100的情况下
'''
svc2 = svm.LinearSVC(C=1000,loss='hinge',max_iter=1000)
svc2.fit(data[['X1','X2']],data['y'])
print(svc2.score(data[['X1','X2']],data['y']))
x1, x2 = find_decision_bundary(svc2, 0, 4, 1.5, 5, 2 * 10**-3)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x1,x2,s=10,label='prediction')
plot_init_data(data,fig,ax)
ax.set_title('SVM (C=100) Decision Boundary')
ax.legend()
plt.show()

def Gaussian(x,y,sigma):
    return np.exp(-np.sum(np.power(x-y,2))/(2*sigma**2))
exp1 = np.array([1,2,1])
exp2 = np.array([0,4,-1])
print(Gaussian(exp1,exp2,sigma=2))