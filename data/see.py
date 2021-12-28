from sklearn.decomposition import PCA
import numpy as np
data = np.load("../data/paviaU_im.npy")
gt = np.load("../data/paviaU_raw_gt.npy")
pca = PCA(n_components=18, svd_solver='randomized')
dataD = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
pca.fit(dataD)
data_D_pca = pca.transform(dataD)
data_PCA = data_D_pca.reshape((data.shape[0], data.shape[1], 18))
print("")
