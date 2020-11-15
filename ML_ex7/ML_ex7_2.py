from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
data = loadmat('D:\python_project\ML_ex7\machine-learning-ex7\ex7\ex7data1.mat')
X = data['X']


def pca(X):
    #归一化非常重要，千万不要忘了
    X = (X-np.mean(X))/np.std(X)
    X = np.matrix(X)
    #算协方差矩阵
    cov =(X.T*X)/X.shape[0]
    #计算SVD
    U,S,V = np.linalg.svd(cov)
    return U,S,V

U,S,V = pca(X)
print(U,S,V)

def project_data(X,U,k):
    '''投影k维数据'''
    U_reduced = U[:,:k]
    return np.dot(X,U_reduced)

#这里的Z就是降维后的数字
Z = project_data(X,U,1)

def recover_data(Z,U,k):
    '''还原原来的维数'''
    U_reduced = U[:,:k]
    return np.dot(Z,U_reduced.T)

X_recover = recover_data(Z,U,1)


fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1],c='b')
ax.scatter(list(X_recover[:,0]), list(X_recover[:,1]),c='r')
plt.show()


'''----------------------分割线---------------------------'''
faces = loadmat('D:\python_project\ML_ex7\machine-learning-ex7\ex7\ex7faces.mat')
X = faces['X']
print(X.shape)

def plot_n_image(X,n):
    '''打印前n个图片'''
    pic_size = int(np.sqrt(X.shape[1]))   #一定要加int转化
    grid_size = int(np.sqrt(n))

    fig,ax = plt.subplots(nrows=grid_size, ncols=grid_size,sharey=True, sharex=True,figsize=(8,8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax[r,c].imshow((np.reshape(X[r*grid_size+c],(pic_size,pic_size))).T, cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

plot_n_image(X,100)



U, S, V = pca(X)
Z = project_data(X,U,100)
X_recover = recover_data(Z,U,100)
plot_n_image(X_recover,100)
plt.show()

fig,ax = plt.subplots(1,2)
face_id = np.reshape(X[3,:],(32,32))
face_id2 = np.reshape(X_recover[3,:],(32,32))
ax[0].imshow(face_id,cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
ax[1].imshow(face_id2,cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()
