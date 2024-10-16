# %%
import numpy as np
import matplotlib.pyplot as plt
import spectral_embedding as se
import matplotlib as mpl
import os
from sklearn.preprocessing import normalize
import scipy.io as sio
np.random.seed(123)

# %%
K = 3
n = 1000
pi = [1/K, 1/K, 1/K]

B0 = np.array([[0.5, 0.3, 0.3], 
                [0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3]])

B1 = np.array([[0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3],
                [0.3, 0.3, 0.3]])

Bs = np.array([B0,B0,B0,B1,B1,B1,B1, B0, B0, B0])

# %%
As, Z = se.generate_SBM_dynamic(n, Bs, pi)
# %%
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:K])
Zcol = colours[Z]

# %%
_, YAs_UASE = se.UASE(As, K)
# %%
T = len(Bs)
# %%
# generate Cs 
p = 150

mu1 = np.repeat(0, p)
mu1[20:75] = 1

mu2 = np.repeat(0, p)
mu2[80:140] = 1
B1# %%
n1 = np.sum(Z==0)
n2 = np.sum(Z==1)
n3 = np.sum(Z==2)
ns = [n1,n2,n3]
# %%
Cs = np.zeros((T, n, p))

for i in range(T):
    C1 = np.random.multivariate_normal(mean = mu1, cov = np.eye(p), size = n1)
    C2 = np.random.multivariate_normal(mean = mu2, cov = np.eye(p), size = n2)
    C3 = np.random.multivariate_normal(mean = mu1, cov = np.eye(p), size = n3)

    C2_middle = np.random.multivariate_normal(mean = mu1, cov = np.eye(p), size = n2)
    Cs[i][Z==0] = C1
    Cs[i][Z==1] = C2
    Cs[i][Z==2] = C3

    if i == 3 or i == 4:
        Cs[i][Z==1] = C2_middle

    # Cs[i] = normalize(Cs[i])
# %%
# save As 
# and Cs

np.save('As.npy', As)
np.save('Cs.npy', Cs)
np.save('Z.npy', Z)

# %%
alpha = 0.2
d = 3
# %%
Acs = []
for i in range(T):
    A = As[i]
    C = Cs[i]
    top = np.hstack([(1-alpha)*A, alpha * C])
    bottom = np.hstack([alpha* C.T, np.zeros((p, p))])
    Ac = np.vstack([top, bottom])
    Acs.append(Ac)

# %%
_, YAs_CUASE = se.UASE(Acs,d)
YAs_CUASE = YAs_CUASE[:,:n,:]

# %%
# %%
YAs_UASE = YAs_UASE.reshape((T*n, K))
YAs_CUASE = YAs_CUASE.reshape((T*n, K))
#%%
# do UMAP
import umap
reducer = umap.UMAP(n_components=1, random_state=0)
#%%
Yt_UASE = reducer.fit_transform(YAs_UASE)
Yt_CUASE = reducer.fit_transform(YAs_CUASE)
#%%
# save Yt_UASE and Yt_CUASE
np.save('Yt_UASE.npy', Yt_UASE)
np.save('Yt_CUASE.npy', Yt_CUASE)
#%%
data_DRLAN = []
for i in range(1,T+1):
    # load embedding_i_03.mat
    data_DRLAN.append(sio.loadmat(f'DRLAN/embedding_{i}beta0.7_option3_d3.mat'))
# %%
# vstack the embeddings
Y_DRLAN = np.vstack([data_DRLAN[i]["U"] for i in range(T)])
# %%
reducer = umap.UMAP(n_components=1, random_state=0)
Yt_DRLAN = reducer.fit_transform(Y_DRLAN)
#%%
# save Yt_DRLAN
np.save('Yt_DRLAN.npy', Yt_DRLAN)

#%%
data_DySAT = []
for i in range(0,T-1):
    # load embedding_i_03.mat
    data_DySAT.append(np.load(f'DySAT/default_embs_figure1_{i}_d3.npz')["data"])
# %%

Yt_DySAT = np.zeros((T-1, n))
for t in range(T-1):
    Yt_DySAT[t] = reducer.fit_transform(data_DySAT[t]).reshape(n)
#%%
# save Yt_DySAT
np.save('Yt_DySAT.npy', Yt_DySAT)
# %%
