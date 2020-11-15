from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize
data = loadmat("D:\python_project\ML_ex8\machine-learning-ex8\ex8\ex8_movies.mat")
Y = data['Y'] #Y（电影数量*用户数量） 表示每个用户给每部电影的分数
R = data['R'] #R（电影数量*用户数量） 表示每个用户是否评分（1表示评了，0表示没有）
print(Y[1,np.where(R[1,:]==1)[0]].mean())

def plot_initdata(Y):
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(Y)
    ax.set_xlabel("users")
    ax.set_ylabel("movies")
    plt.tight_layout() #加不加无所谓，自动调整子图参数,使之填充整个图像区域
    plt.show()

param_data = loadmat("D:\python_project\ML_ex8\machine-learning-ex8\ex8\ex8_movieParams.mat")
X = param_data['X']
theta = param_data['Theta']
print(X.shape, theta.shape)  #X(电影数量*特征数量） theta(用户数量*特征数量）

def serialize(X,theta):
    '''序列化，因为参数学习时不能传两个，所以要将其合并'''
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    #该函数完成多个数组的拼接(注意传入是一个元组）
    return np.concatenate((X.ravel(), theta.ravel()))

def deserialize(param, movies, users, features):
    '''逆序列化'''
    return param[:movies*features].reshape(movies,features), param[movies*features:].reshape(users,features)

def cost(param, Y, R, features):
    """compute cost for every r(i, j)=1
        Args:
            param: serialized X, theta
            Y (movie, user), (1682, 943): (movie, user) rating
            R (movie, user), (1682, 943): (movie, user) has rating
        """
    n_movies, n_users = Y.shape
    X,theta = deserialize(param, n_movies, n_users,features)
    cost = np.multiply(np.power(X@theta.T-Y,2),R)
    return np.sum(cost)/2

print(cost(serialize(X,theta),Y,R,10))


def gradient(param, Y, R, features):
    '''生成梯度，注意最后依然是一个单独的值'''
    n_movies, n_users = Y.shape
    X, theta = deserialize(param, n_movies, n_users, features)
    cost = np.multiply(X @ theta.T - Y, R)
    #cost (1682, 943) X (1682, 10) theta(943,10)
    X_grad = cost@theta
    theta_grad = cost.T@X
    return serialize(X_grad,theta_grad)


n_movies, n_users = Y.shape
print(deserialize(gradient(serialize(X,theta),Y,R,10),n_movies,n_users,10))

def regularized_cost(param, Y, R, features,punishing=1):
    """加上正则化后的误差函数"""
    punish = np.sum(np.power(param,2))*punishing/2
    return cost(param,Y,R,features)+punish

def regularized_gradient(param, Y, R, features,punishing=1):
    """加上正则化的梯度值"""
    punish = punishing*param

    return gradient(param,Y,R,features)+punish

print(regularized_cost(serialize(X,theta),Y,R,10))

print(deserialize(regularized_gradient(serialize(X,theta),Y,R,10),n_movies,n_users,10))

file_path ="D:\python_project\ML_ex8\machine-learning-ex8\ex8\movie_ids.txt"
movie_list=[]
f = open(file_path, encoding='utf-8')
for line in f:
    tokens = line.strip().split(' ')
    movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)
print(movie_list[0])

'''自己的评价指标'''
ratings = np.zeros((1682, 1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5
"""自己称为第0个user"""
Y = np.append(ratings,Y,axis=1)
R = np.append(ratings!=0, R, axis=1)

n_movies, n_users = Y.shape
punishing = 10
n_features = 10
X = np.random.random((n_movies,n_features))
theta = np.random.random((n_users,n_features))
param = serialize(X,theta)
Y_norm = Y - Y.mean()

fmin = minimize(regularized_cost, x0=param, args=( Y, R, n_features,punishing),method='TNC',
                jac=regularized_gradient)
print(fmin)

X_trained, theta_trained = deserialize(fmin.x,n_movies,n_users,n_features)
print(X_trained.shape, theta_trained.shape)

all_predict = X_trained@theta_trained.T  #(1682, 944)
my_predict = all_predict[:,0]+Y.mean()
my_favourite = np.argsort(my_predict)[::-1]   #逆序排练（从大到小）。返回的是my_predict从大到校的索引值
for element in my_predict[my_favourite][:10]:
    print( element)
for element in movie_list[my_favourite][:10]:
    print(element)

