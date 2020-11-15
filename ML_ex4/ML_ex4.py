import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report

'''数据导入'''
path1 = 'D:\python_project\ML_ex4\machine-learning-ex4\ex4\ex4data1.mat'
path2 = 'D:\python_project\ML_ex4\machine-learning-ex4\ex4\ex4weights.mat'
data = loadmat(path1)
weight = loadmat(path2)
X, y = data['X'], data['y']
theta1, theta2 = weight['Theta1'], weight['Theta2']
print(X.shape, y.shape, theta1.shape, theta2.shape)


'''100个样本进行可视化，显示画面'''
sample_idx = np.random.choice(X.shape[0],100,replace=False)  #这是np.random，而不是random里的函数
sample_X = X[sample_idx,:]
print(sample_X.shape)
fig, ax = plt.subplots(figsize=(12,12),sharex=True, sharey=True, ncols=10, nrows=10)
for i in range(10):
    for j in range(10):
        ax[i,j].matshow(sample_X[i*10+j,:].reshape(20,20).T,cmap=matplotlib.cm.binary)  #是ax[i,j]，而不是ax
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()


def sigmoid_func(z):
    return 1/(np.exp(-z)+1)

'''前向传播函数'''
def forward_propagate(X, theta1, theta2):
    a1 = np.insert(X, 0, values=1, axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(sigmoid_func(z2), 0, values=1, axis=1)
    z3 = a2*theta2.T
    h = sigmoid_func(z3)
    return a1, z2, a2, z3, h

#forward_propagate(X,theta1, theta2)

'''计算误差函数J'''
def cost(theta1, theta2, X, y):
    m = X.shape[0] #样本数量
    X, theta1, theta2, y  = np.matrix(X), np.matrix(theta1), np.matrix(theta2), np.matrix(y)
    a1, z2, a2, z3, h = forward_propagate(X,theta1, theta2)
    '''后面的代码也可以写成for函数'''
    '''
    J = 0
    for i in range(m):
        first = np.multiply(-y[i,:],np.log(h[i,:]))
        second = np.multiply(1-y[i,:],np.log(1-h[i,:]))
        J += np.sum(first-second)
    return J/m
    '''
    first = np.multiply(-y,np.log(h))
    second = np.multiply(1-y, np.log(1-h))
    return np.sum(first-second)/m

'''onehot编码'''
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)
'''计算起始误差'''
print(cost(theta1, theta2,X,y_onehot))


'''正则化代价函数'''
def costReg(theta1, theta2, X, y, punishingrate):
    cost1 = cost(theta1,theta2, X,y)
    m = X.shape[0]
    cost2 = punishingrate/(2*m)*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
    return cost1+cost2
punishingrate = 1
print(costReg(theta1,theta2,X,y_onehot,punishingrate))

'''
以上都是正向传播的一些内容，
下面涉及反向传播算法
'''

'''sigmoid梯度，sigmoid求导'''
def sigmoid_grad(z):
    return np.multiply(1-sigmoid_func(z),sigmoid_func(z))

'''随机赋初值'''
eposilon = 0.12
#设置随机值的数量为两层theta的size,后面在重新截取
params = (np.random.random(size=theta1.shape[0]*theta1.shape[1]+theta2.shape[0]*theta2.shape[1])-0.5)*2*eposilon


'''反向传播函数算法'''
def backward_propagate(params, X, y, theta1, theta2,  punishingrate):
    X = np.matrix(X)
    y = np.matrix(y)
    m = X.shape[0]

    #随机值的分配，原先的theta1和theta2只起到确定形状的作用
    t1s1, t1s2, t2s1, t2s2 = theta1.shape[0], theta1.shape[1], theta2.shape[0], theta2.shape[1]
    theta1 = np.matrix(np.reshape(params[:t1s1*t1s2],(t1s1,t1s2)))  #25*401
    theta2 = np.matrix(np.reshape(params[t1s1*t1s2:],(t2s1,t2s2)))  #10*26
    print(theta1.shape,theta2.shape)

    '''执行新的一轮前向随机算法'''
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = cost(theta1,theta2,X,y)
    print(J)

    '''定义每层的神经元delta'''
    delta1 = np.zeros(theta1.shape)  #25*401
    delta2 = np.zeros(theta2.shape)  #10*26

    for t in range(m):
        a1t = a1[t,:]   #1*401
        z2t = z2[t,:]   #1*25
        a2t = a2[t,:]   #1*26
        z3t = z3[t,:]   #1*10
        ht = h[t,:]     #1*10
        yt = y[t,:]     #1*10

        d3 = ht-yt  #1*10
        z2t = np.insert(z2t,0,values=np.ones(1))  #1*26
        d2 = np.multiply((theta2.T*d3.T).T,sigmoid_grad(z2t))    #1*26

        delta1  = delta1 + d2[:,1:].T*a1t
        delta2 = delta2 + d3.T*a2t
    delta1 = delta1/m
    delta2 = delta2/m
    return J,delta1,delta2 




def backward_propReg(params, X, y, theta1, theta2, punishingrate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 随机值的分配，原先的theta1和theta2只起到确定形状的作用
    t1s1, t1s2, t2s1, t2s2 = theta1.shape[0], theta1.shape[1], theta2.shape[0], theta2.shape[1]
    theta1 = np.matrix(np.reshape(params[:t1s1 * t1s2], (t1s1, t1s2)))  # 25*401
    theta2 = np.matrix(np.reshape(params[t1s1 * t1s2:], (t2s1, t2s2)))  # 10*26
    print(theta1.shape, theta2.shape)

    '''执行新的一轮前向随机算法'''
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = costReg(theta1, theta2, X, y,punishingrate)
    print(J)

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for t in range(m):
        a1t = a1[t, :]  # 1*401
        z2t = z2[t, :]  # 1*25
        a2t = a2[t, :]  # 1*26
        z3t = z3[t, :]  # 1*10
        ht = h[t, :]  # 1*10
        yt = y[t, :]  # 1*10

        d3t = ht-yt  #1*10
        z2t = np.insert(z2t,0, values=1)
        d2t = np.multiply((d3t*theta2), sigmoid_grad(z2t))  #1*26

        delta1 = delta1 + d2t[:,1:].T*a1t
        delta2 = delta2 + d3t.T*a2t
    delta1 = delta1/m  #25*401
    delta2 = delta2/m  #10*26

    '''加上正则化参数'''
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:]*punishingrate/m)
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:]*punishingrate/m)
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    #concatenate与append类似，但其效率更高，去掉array的最外层括号【】，进行拼接，再添上最外层括号【】
    print(grad.shape)
    return J, grad

##最优化参数
#fun就是反向学习算法函数，X0就是要调节的参变量，args就是剩下的所有形参变量
fmin = minimize(fun=backward_propReg,x0=params, args=(X,y_onehot,theta1,theta2,punishingrate),
                method='TNC',jac=True,options={'maxiter':250})
print(fmin)

X = np.matrix(X)
theta1_f = np.matrix(np.reshape(fmin.x[:theta1.shape[0]*theta1.shape[1]],(theta1.shape[0],theta1.shape[1])))
theta2_f = np.matrix(np.reshape(fmin.x[theta1.shape[0]*theta1.shape[1]:],(theta2.shape[0],theta2.shape[1])))

a1_f, z2_f, a2_f, z3_f, h_f = forward_propagate(X,theta1_f,theta2_f)
y_pre = np.array(np.argmax(h_f,axis=1)+1)
print(y_pre)
print(classification_report(y,y_pre))

'''可视化隐藏层'''
hidden_layer = theta1_f[:,1:]  #25*400

fig,ax= plt.subplots(sharex=True, sharey=True, ncols=5, nrows=5,figsize=(12,12))
for i in range(5):
    for j in range(5):
        ax[i,j].matshow(np.array(np.reshape(hidden_layer[i*5+j,:],(20,20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()






