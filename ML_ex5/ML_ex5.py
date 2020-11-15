from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import pandas as pd
path = 'D:\python_project\ML_ex5\machine-learning-ex5\ex5\ex5data1.mat'
data = loadmat(path)
X, y, Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']
print(X.shape, y.shape, Xtest.shape, ytest.shape, Xval.shape, yval.shape)
###这里是用map函数实现一个函数同时作用多个变量，这里用np.ravel同时作用在六个变量，实现一维化
X, y, Xtest, ytest, Xval, yval = map(np.ravel,(X, y, Xtest, ytest, Xval, yval) )
print(X.shape, y.shape, Xtest.shape, ytest.shape, Xval.shape, yval.shape)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X,y,c='r',marker='x')
ax.set_xlabel("water_level")
ax.set_ylabel("flow")
#plt.show()

'''一个统一的写法！！！！'''
X, Xtest, Xval =[np.insert(np.reshape(x,(x.shape[0],1)),0,values=1,axis=1) for x in (X, Xtest, Xval)]  #这里一定注意，因为X原来是一维
y, ytest, yval = [np.reshape(x,(x.shape[0],1)) for x in (y, ytest, yval)] #这句话一定要写，否则后面y会当成1*12
print(X.shape, y.shape, Xtest.shape, ytest.shape, Xval.shape, yval.shape)


'''四件套'''
def cost(theta,X,y):
    X = np.matrix(X)
    theta = np.matrix(theta)
    y = np.matrix(y)
    m = X.shape[0]
    h = X*theta.T
    return np.sum(np.power(h-y,2))/(2*m)

def costReg(theta, X,y,punishingrate):
    m = X.shape[0]
    punish = punishingrate* np.sum(np.power(theta[1:],2))/(2*m)
    return punish+cost(theta, X,y)

def gradient(theta,X,y):
    m = X.shape[0]
    X = np.matrix(X)
    theta = np.matrix(theta)
    y = np.matrix(y)
    h = X*theta.T-y
    return np.ravel((X.T*h).T/m)

def gradientReg(theta,X,y,punishingrate):
    m = X.shape[0]
    grad = gradient(theta,X,y)
    #grad.shape ==1*2
    grad[1:] += punishingrate * theta[1:]/m
    return np.ravel(grad)

punishingrate = 1
theta = np.array([1,1])
print(costReg( theta,X, y,punishingrate))
print(gradient(theta,X,y),gradientReg(theta, X,y,punishingrate))

'''拟合线性回归'''
'''这里将lambda设为0'''
final_theta = minimize(costReg,x0=theta, args=(X,y,0),method='TNC',jac=gradientReg).x
print(final_theta)

x0 = np.linspace(-45,45,100)
y0 = x0*final_theta[1]+final_theta[0]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,1],y,c='r',marker='x',label='training data')
ax.plot(x0,y0,label='prediction')
ax.set_xlabel("water_level")
ax.set_ylabel("flow")
#plt.show()

def trainLinearReg(theta,X,y,punishingrate):
    """
    inear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
        return: trained parameters
    """
    res = minimize(costReg,x0=theta, args=(X,y,punishingrate),method='TNC',jac=gradientReg,options={'disp': True}).x
    return res

training_cost=[]
cv_cost=[]

for i in range(1,X.shape[0]+1):     #这里一定是1到X.shape[0]+1，注意这个bug
    res = trainLinearReg(theta, X[:i, :],  y[:i, :],  0)
    train_error = cost(res, X[:i, :],  y[:i, :])
    cv_error = cost(res, Xval, yval)
    training_cost.append(train_error)
    cv_cost.append(cv_error)

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1,X.shape[0]+1), training_cost, c='b', label='training data')
ax.plot(np.arange(1,X.shape[0]+1) , cv_cost, c='r', label="cv data")
ax.legend()
#plt.show()

''''
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
多项式回归
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
'''

####多项式变化，采用次方项
def polyFeatures(X,power,as_nddarray=False):
    data = {'f{:}'.format(i) : np.power(X,i) for i in range(1,power+1)}
    df = pd.DataFrame(data)

    return df.values if as_nddarray else df

X, y, Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']
X, y, Xtest, ytest, Xval, yval = map(np.ravel,(X, y, Xtest, ytest, Xval, yval) )
print(polyFeatures(X,3))  #如果是True，则返回一个array形式

'''重点：！！！一定要归一化'''
def normalization(df):
    ''' 默认情况下 apply参数中axis为0 or 'index': apply function to each column '''
    return df.apply(lambda colomn: (colomn-colomn.mean())/colomn.std() )

def prep_polyFeatures(*args, power):
    '''
    对于参数arg中的每一个值，都进行一次操作
    因为 *args可以看成一个可变的参数，传递进来时一个元组的形式
    '''
    ##下面是一个内置的函数
    def prepare(x,power):
        data1 = polyFeatures(x,power)
        data1 = normalization(data1).values
        data1 = np.insert(data1,0,values=1,axis=1)
        return data1
    return [prepare(x,power) for x in args]

X_poly, Xtest_poly, Xval_poly = prep_polyFeatures(X, Xtest, Xval, power=8)
print(X_poly[:3,:])

y_poly, ytest_poly, yval_poly = [np.reshape(x,(x.shape[0],1)) for x in (y, ytest, yval)] #这句话一定要写，否则后面y会当成1*12



'''绘制多项式的学习曲线，这里先不采用正则化'''
def plot_poly_learning_curve(X, X_poly, y_poly, Xval_poly, yval_Poly, punishingrate,power):
    theta = np.ones(X_poly.shape[1])
    training_cost = []
    cv_cost = []
    for i in range(1,X_poly.shape[0]+1):
        train_errors=0
        cv_errors=0
        for j in range(50):
            set1 = np.random.choice(X_poly.shape[0], i)
            set2 = np.random.choice(X_poly.shape[0], i)
            res = trainLinearReg(theta, X_poly[set1, :], y_poly[set1, :], punishingrate)
            train_errors += cost(res,X_poly[set1,:],y_poly[set1,:])
            cv_errors += cost(res, Xval_poly[set2,:], yval_Poly[set2,:])
        training_cost.append(train_errors/50)
        cv_cost.append(cv_errors/50)

    print(training_cost,cv_cost)
    '''绘制学习曲线'''
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(12,12))
    ax[0].plot(range(1,X_poly.shape[0]+1), training_cost, c='b', label="training data")
    ax[0].plot(range(1,X_poly.shape[0]+1),cv_cost, c='r',label="cv data")
    ax[0].legend()
    '''绘制最后一次训练集上的拟合曲线'''
    x = np.linspace(-45,45,100)
    y = np.dot(prep_polyFeatures(x,power=power), trainLinearReg(theta,X_poly,y_poly,punishingrate))
    y = np.ravel(y)
    ax[1].scatter(X,y_poly,marker='x',label='training data')
    ax[1].plot(x,y,label='prediction')
    ax[1].legend()
    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')
    plt.show()

plot_poly_learning_curve(X,X_poly,y_poly,Xval_poly,yval_poly,0.01,8)

def punishingrate_curve(X_poly,y_poly,Xval_poly, yval_poly):
    punishingrates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    theta = np.ones(X_poly.shape[1])
    training_cost = []
    cv_cost = []
    for punishingrate in punishingrates:
        res = trainLinearReg(theta, X_poly, y_poly, punishingrate)
        train_error = cost(res,X_poly, y_poly)
        cv_error = cost(res, Xval_poly, yval_poly)
        training_cost.append(train_error)
        cv_cost.append(cv_error)
    fig,ax = plt.subplots(figsize=(12,8))
    ax.plot(punishingrates, training_cost, c='b', label='training data')
    ax.plot(punishingrates, cv_cost, c='r', label='cv data')
    ax.set_xlabel("lambda")
    ax.legend()
    plt.show()

#punishingrate_curve(X_poly,y_poly,Xval_poly, yval_poly)

punishingrates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for i in punishingrates:
    theta = np.ones(X_poly.shape[1])
    res = trainLinearReg(theta, X_poly, y_poly, i)
    print("test cost({}):  =  ".format(i)+str(cost(res,Xtest_poly,ytest_poly)))
