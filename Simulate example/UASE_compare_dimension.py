#%%
import numpy as np
import matplotlib.pyplot as plt
import spectral_embedding as se
import matplotlib as mpl
import os
from sklearn.preprocessing import normalize
import scipy.io as sio
import umap
np.random.seed(123)
#%%
# load As 
K = 3
n = 1000
pi = [1/K, 1/K, 1/K]
T =10
p = 150
#%%
As = np.load('As.npy')
Cs = np.load('Cs.npy')
Z = np.load('Z.npy')
#%%
#%%
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:K])
#%%
Z_new = np.zeros(n)
Z_new[Z==1] = int(2)
Z_new[Z==2] = int(1)
Z = Z_new
Z = Z.astype(int)
Zcol = colours[Z]
#%% 
ds = [16,32,64]
Yt_UASE_t_means = []
Yt_UASE_t_lowers = []
Yt_UASE_t_uppers = []
alpha =0.2
for t,d in enumerate(ds):

    _, YAs_UASE = se.UASE(As,d)
    YAs_UASE = YAs_UASE[:,:n,:]

    YAs_UASE = YAs_UASE.reshape((T*n, d))

    reducer = umap.UMAP(n_components=1, random_state=0)

    Yt_UASE = reducer.fit_transform(YAs_UASE)

    Yt_UASE_t =np.array( np.array_split(Yt_UASE, T)).reshape((T, n))
    # compute the mean for each value of Z
    Yt_UASE_t_means.append([np.mean(Yt_UASE_t[:,Z==i], axis = 1) for i in range(K)])
    Yt_UASE_t_lowers.append([np.percentile(Yt_UASE_t[:,Z==i], 5, axis =1) for i in range(K)])
    Yt_UASE_t_uppers.append([np.percentile(Yt_UASE_t[:,Z==i], 95, axis =1) for i in range(K)])
    
# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 20))

for i in range(K):
    axs[0].plot(range(T), Yt_UASE_t_means[0][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0].plot(T, 3, color = 'white')
    axs[0].fill_between(range(T), Yt_UASE_t_lowers[0][i], 
                          Yt_UASE_t_uppers[0][i], color = colours[i], alpha=0.11)
axs[0].axvline(x=2.5, color='k', linestyle='--')
axs[0].axvline(x=4.5, color='k', linestyle='--')
axs[0].axvline(x=6.5, color='k', linestyle='--')
axs[0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0].set_title('d=16', fontsize = 11)

for i in range(K):
    axs[1].plot(range(T), Yt_UASE_t_means[1][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1].plot(T, 3, color = 'white')
    axs[1].fill_between(range(T), Yt_UASE_t_lowers[1][i], 
                          Yt_UASE_t_uppers[1][i], color = colours[i], alpha=0.11)
axs[1].axvline(x=2.5, color='k', linestyle='--')
axs[1].axvline(x=4.5, color='k', linestyle='--')
axs[1].axvline(x=6.5, color='k', linestyle='--')
axs[1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1].set_title('d=32', fontsize = 11)

for i in range(K):
    axs[2].plot(range(T), Yt_UASE_t_means[2][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[2].plot(T, 3, color = 'white')
    axs[2].fill_between(range(T), Yt_UASE_t_lowers[2][i], 
                          Yt_UASE_t_uppers[2][i], color = colours[i], alpha=0.11)
axs[2].axvline(x=2.5, color='k', linestyle='--')
axs[2].axvline(x=4.5, color='k', linestyle='--')
axs[2].axvline(x=6.5, color='k', linestyle='--')
axs[2].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[2].set_title('d=128', fontsize = 11)
# save the plot
plt.savefig('UASE_dimensions.png')
# %%
