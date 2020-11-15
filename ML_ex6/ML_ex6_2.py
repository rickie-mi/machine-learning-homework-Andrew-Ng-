import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb   #提供了更好的可视化，相比于matplotlib
from scipy.io import loadmat
from sklearn import svm

'''简单模拟一下高斯核函数'''
def Gaussian(x,y,sigma):
    return np.exp(-np.sum(np.power(x-y,2))/(2*sigma**2))
exp1 = np.array([1,2,1])
exp2 = np.array([0,4,-1])
print(Gaussian(exp1,exp2,sigma=2))

def plot_init_data(data,fig,ax):
    positive = data[data['y'].isin(['1'])]
    negative = data[data['y'].isin(['0'])]
    ax.scatter(positive['X1'], positive['X2'],c='r',marker='x')
    ax.scatter(negative['X1'], negative['X2'],c='b',marker='o')

path = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\ex6data2.mat'
init_data = loadmat(path)
data = pd.DataFrame(init_data['X'],columns=['X1','X2'])
data['y'] = init_data['y']
fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data,fig,ax)
ax.legend()
plt.show()

'''使用高斯核函数'''

'''
C的变化对于误差的影响 -----------可以发现随着C的增大，惩罚系数变大，正确率越高
accuracys = []
for C in np.linspace(0.01,1,100):
    svc = svm.SVC(C=C, kernel='rbf', gamma=10, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    accuracys.append(svc.score(data[['X1','X2']],data['y']))
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.linspace(0.01,1,100),accuracys)
ax.set_xlabel("C")
ax.set_ylabel('accuracy')
plt.show()
'''

'''
gamma的变化对于误差的影响 -----------可以发现随着gamma的增大，正确率越高
accuracys = []
for gamma in np.linspace(1,100,100):
    svc = svm.SVC(C=1, kernel='rbf', gamma=gamma, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    accuracys.append(svc.score(data[['X1','X2']],data['y']))
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.linspace(1,100,100),accuracys)
ax.set_xlabel("gamma")
ax.set_ylabel('accuracy')
plt.show()
'''

#这里选取C=100，gamma为10的情况
svc = svm.SVC(C=100,gamma=10,kernel='rbf',probability=True)
svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1','X2']],data['y']))

def find_decision_boundary(svc, x1min, x1max, x2min, x2max,differ):
    x1 = np.linspace(x1min,x1max,1000)
    x2 = np.linspace(x2min,x2max,1000)
    corderate = [(x,y) for x in x1 for y in x2]
    corderate1, corderate2 = zip(*corderate)
    f_val = pd.DataFrame({'x1':corderate1, 'x2':corderate2})
    f_val['val'] = svc.decision_function(f_val[['x1','x2']])
    decision = f_val[np.abs(f_val['val'])<differ]
    return decision.x1, decision.x2

x1,x2 = find_decision_boundary(svc,0,1,0.4,1,2*10**-3)

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(x1,x2,s=5)
plot_init_data(data,fig,ax)
ax.legend()
plt.show()