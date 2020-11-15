import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import seaborn as sb

data = loadmat('D:\python_project\ML_ex7\machine-learning-ex7\ex7\ex7data2.mat')
X = data['X']  # 300个二维数据


def init_centroid(X, k):
    """随机选取k个样本点作为聚类中心点"""
    m, n = X.shape
    init = np.zeros((k, n))
    idx = np.random.randint(0, m, k)  # 随机选取k个值在0~m之间的值

    for i in range(k):
        init[i, :] = X[idx[i], :]
    return init


def find_closet_centroid(X, centroids):
    '''第一步骤：寻找每个点所属于的类别是哪个'''
    points = X.shape[0]
    sorts = centroids.shape[0]
    res = np.zeros(points)
    for point in range(points):
        minsort, minlen = -1, 10000000
        for sort in range(sorts):
            if np.sum(np.power((X[point] - centroids[sort]), 2)) < minlen:
                minlen = np.sum(np.power((X[point] - centroids[sort]), 2))
                minsort = sort
        res[point] = minsort
    return res


def compute_centroids(X, idx):
    '''第二步骤，更新聚类中心点'''
    k = len(np.unique(idx))
    m, n = X.shape[0], X.shape[1]
    res = np.zeros((k, n))
    for i in range(k):  # 外循环只需循环分类总数
        indices = np.where(idx == i)  # 返回索引,类型是一个tuple类型
        res[i, :] = np.sum(X[indices, :], axis=1) / len(indices[0])  # 这里需要加axis=1,否则是把所有和加起来了，indices是二维的
    return res


def run_k_means(X, k, iterations):
    '''k-means算法'''
    m, n = X.shape[0], X.shape[1]
    '''随机定义三个点'''
    # centroids = np.array([[3, 3], [6, 2], [8, 5]])
    centroid_history = []
    centroids = init_centroid(X, k)
    print(centroids)
    idx = np.zeros(m)
    for i in range(iterations):
        idx = find_closet_centroid(X, centroids)
        centroids = compute_centroids(X, idx)
        centroid_history.append(centroids)
    return centroids, idx, centroid_history


'''第一种呈现数据的方式'''
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
'''第二种呈现数据的方式'''
df = pd.DataFrame(data['X'], columns=['X1', 'X2'])
sb.set(context='notebook', style='white')
sb.lmplot('X1', 'X2', data=df, fit_reg=False)
# print(df.head())

# plt.show()


'''
    测试时使用
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closet_centroid(X,initial_centroids)
print(idx[:3])
centroids = compute_centroids(X,idx)
print(centroids)

'''
final_centroids, final_idx, centroid_history = run_k_means(X, 3, 10)
cluster1 = X[np.where(final_idx == 0)[0], :]  # 这里必须加上[0]
cluster2 = X[np.where(final_idx == 1)[0], :]
cluster3 = X[np.where(final_idx == 2)[0], :]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], c='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], c='b', label='Cluster 1')
ax.scatter(cluster3[:, 0], cluster3[:, 1], c='g', label='Cluster 1')
ax.legend()
plt.show()

'''----------------------分割线---------------------------'''

from IPython.display import Image

Image(filename='D:\python_project\ML_ex7\machine-learning-ex7\ex7\_bird_small.png')

image_data = loadmat('D:\python_project\ML_ex7\machine-learning-ex7\ex7\_bird_small.mat')
A = image_data['A']
print(A.shape)  # 原始尺寸128*128*3
##对A进行预处理,归一化，同时横纵坐标统一一下
plt.imshow(A)
plt.show()
A = A / 255
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)

'''将画面中的值转化为16个类别'''
image_centroids, image_idx, centroid_history = run_k_means(X, 16, 10)
'''将这些点根据划分的聚类结果，用聚类中心点来表示其颜色情况'''
X_recover = image_centroids[image_idx.astype(int), :]  # 这里image_idx中都是小数。例如1表示为1.
X_recover = np.reshape(X_recover, (A.shape[0], A.shape[1], A.shape[2]))
plt.imshow(X_recover)  # imshow可以接收0-1或者0-255
plt.show()

'''
空间分析，原来图片中有128*128个点，每个点用rgb表示，其中rgb的值为0-255（8bits)
所以总的空间占用是128*128*3*8bits
压缩过后，由于只需要16个颜色点就可以表示，所以内存中需要存储16个颜色的空间：16*3*8bits
其余的点只需要用这16个点表示，也就是0~15（4bits)
所以总的空间大小为16*3*8+128*128*4bits
'''

'''调用现成的库scikit-learn来实现k-means算法'''
from skimage import io

pic = io.imread('D:\python_project\ML_ex7\machine-learning-ex7\ex7\_bird_small.png') / 255
print(pic.shape)  # 128*128*3
data = pic.reshape(128 * 128, 3)
from sklearn.cluster import KMeans

model = KMeans(n_clusters=32, n_init=100, n_jobs=-1)
model.fit(data)
centroids = model.cluster_centers_  # [32,3]代表32个点
C = model.predict(data)  # [128*128]代表属于哪个类
comprised_pic = centroids[C].reshape((128, 128, 3))
fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
ax[0].imshow(pic)
ax[1].imshow(comprised_pic)
plt.show()
