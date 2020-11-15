import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
'''导入数据'''
path = 'D:\python_project\ML_ex2\machine-learning-ex2\ex2\ex2data1.txt'
data = pd.read_csv(path,header=None,names=['exam1','exam2','adm'])
'''区分分类的类别'''
postive = data[data['adm'].isin([1])]     #会筛选出adm为1或者0的数据，并给相应的变量
negative = data[data['adm'].isin([0])]

def plot_data(postive,negative):
    '''绘制图像'''
    fig,ax = plt.subplots(figsize=(12,8))
    ax.scatter(postive['exam1'],postive['exam2'],s=50,c='b',marker='o',label='admitted')
    ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='turned down')
    ax.legend()  #显示图例，也就是label值
    ax.set_xlabel('exam1')
    ax.set_ylabel('exam2')
    plt.show()

##plot_data(postive,negative)

def sigmoid_func(z):
    '''定义sigmoid函数'''
    return 1/(1+np.exp(-1*z))

def cost_func(theta,X,y):   #y是m*1, X是m*(n+1)，theta是1*(n+1)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    '''-y*ln(h)-(1-y)*ln(1-h)'''
    '''np.multiply表示对应位置相乘'''
    first = np.multiply(-y,np.log(sigmoid_func(X*theta.T)))
    second = np.multiply(-(1-y),np.log(1-sigmoid_func(X*theta.T)))
    return sum(first+second)/len(y)

'''获得数据'''
data.insert(0,'ones',1)
cols = data.shape[1]   #获取列数
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)
X = np.array(X)   #这里的X和y都是dataframe类型，转化为numpy.array类型
y = np.array(y)
print(X.shape,y.shape,theta.shape)
print(cost_func(theta,X,y))

'''获取梯度函数'''
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = theta.shape[1]   #获取参数个数
    grad = np.zeros(parameters)
    error = sigmoid_func(X*theta.T)-y

    for i in range(parameters):
        term = np.multiply(error,X[:,i])
        grad[i] = sum(term)/len(y)
    return grad

#用自带函数进行优化
result = opt.fmin_tnc(func=cost_func,x0=theta,fprime=gradient,args=(X,y))
print(result)
print(cost_func(result[0],X,y))


def draw_prediction(result,data):
    '''分割线就是h等于0的那条线，即theta[0]+theta[1]*x1+theta[2]*x2=0'''
    plot_x = np.linspace(30,100,100)
    plot_y = (-result[0][0]-plot_x*result[0][1])/result[0][2]
    fig,ax = plt.subplots(figsize=(12,8))
    ax.plot(plot_x,plot_y,'y',label='prediction')
    ax.scatter(postive['exam1'], postive['exam2'], s=50, c='b', marker='o', label='admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='turned down')
    ax.legend()
    ax.set_xlabel('exam1')
    ax.set_ylabel('exam2')
    plt.show()
draw_prediction(result,data)

def hfunc1(theta,X):
    return sigmoid_func(np.dot(theta,X))  #返回对应位置相乘后的相加

'''预测x1=45,x2=85通过的概率'''
print(hfunc1(result[0],[1,45,85]))


def prediction_training(theta,X):
    '''将模型运用到训练集上判断'''
    #pro = hfunc1(X,theta.T) 写成这个形式也行
    pro =sigmoid_func(X*theta.T)
    return [1 if x>=0.5 else 0 for x in pro]  #注意这种写法

def judge_accuracy(theta,X,y):
    result_mat = np.matrix(result[0])  # 必须转化为矩阵形式
    pros = prediction_training(result_mat, X)
    accuy=0
    res=[]
    for i in range(len(pros)):
        if pros[i]==y[i]:
            accuy+=1
            res.append(1)
        else:
            res.append(0)
    print(accuy/len(pros))
    return res
print(judge_accuracy(result[0],X,y))
