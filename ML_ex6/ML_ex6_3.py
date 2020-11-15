import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb   #提供了更好的可视化，相比于matplotlib
from scipy.io import loadmat
from sklearn import svm

path = 'D:\python_project\ML_ex6\machine-learning-ex6\ex6\ex6data3.mat'
raw_data = loadmat(path)
X_val = raw_data['Xval']
y_val = raw_data['yval']
X = raw_data['X']
y = raw_data['y']
data = pd.DataFrame(X,columns=['X1','X2'])
data['y'] = y

def plot_init_data(data,fig,ax):
    positive = data[data['y'].isin(['1'])]
    negative = data[data['y'].isin(['0'])]
    ax.scatter(positive['X1'], positive['X2'],c='r',marker='x')
    ax.scatter(negative['X1'], negative['X2'],c='b',marker='o')

def find_decision_boundary(svc, x1min, x1max, x2min, x2max,differ):
    x1 = np.linspace(x1min,x1max,1000)
    x2 = np.linspace(x2min,x2max,1000)
    corderate = [(x,y) for x in x1 for y in x2]
    corderate1, corderate2 = zip(*corderate)
    f_val = pd.DataFrame({'x1':corderate1, 'x2':corderate2})
    f_val['val'] = svc.decision_function(f_val[['x1','x2']])
    decision = f_val[np.abs(f_val['val'])<differ]
    return decision.x1, decision.x2

fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data,fig,ax)
ax.legend()
#plt.show()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_param ={'C':None, 'gamma':None}
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C,gamma=gamma,probability=True)
        svc.fit(X,y)
        score = svc.score(X_val,y_val)
        if score>best_score:
            best_score = score
            best_param['C'] = C
            best_param['gamma'] = gamma

print(best_param, best_score)

svc = svm.SVC(C=best_param['C'], gamma=best_param['gamma'],probability=True)
svc.fit(X,y)
x1,x2 = find_decision_boundary(svc,-0.6,0.2,-0.6,0.6,2*10**-3)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x1,x2,s=5)
plot_init_data(data,fig,ax)
plt.show()


