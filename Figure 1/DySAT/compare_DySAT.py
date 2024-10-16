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
Z = np.load('../Z.npy')
#%%
# DySAT embeddings for d= 3, 16, 32, 64 in the same directory
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
Yt_DySAT_t_means = []
Yt_DySAT_t_lowers = []
Yt_DySAT_t_uppers = []
for t,d in enumerate(ds):
    data_DySAT = []
    for i in range(0,T-1):
        # load embedding_i_03.mat
        data_DySAT.append(np.load(f'default_embs_figure1_{i}_d{d}.npz')["data"])

    Yt_DySAT = np.zeros((T-1, n))

    reducer = umap.UMAP(n_components=1, random_state=0)
    for t in range(T-1):
        Yt_DySAT[t] = reducer.fit_transform(data_DySAT[t]).reshape(n)
    # compute the mean for each value of Z
    Yt_DySAT_t_means.append( [np.mean(Yt_DySAT[:,Z==i], axis = 1) for i in range(K)])
    Yt_DySAT_t_lowers.append([np.percentile(Yt_DySAT[:,Z==i], 5, axis =1) for i in range(K)])
    Yt_DySAT_t_uppers.append([np.percentile(Yt_DySAT[:,Z==i], 95, axis =1) for i in range(K)])
    
# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 20))

for i in range(K):
    axs[0].plot(range(T-1), Yt_DySAT_t_means[0][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0].plot(T, 3, color = 'white')
    axs[0].fill_between(range(T-1), Yt_DySAT_t_lowers[0][i], 
                          Yt_DySAT_t_uppers[0][i], color = colours[i], alpha=0.11)
axs[0].axvline(x=2.5, color='k', linestyle='--')
axs[0].axvline(x=4.5, color='k', linestyle='--')
axs[0].axvline(x=6.5, color='k', linestyle='--')
axs[0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0].set_title('d=16', fontsize = 11)

for i in range(K):
    axs[1].plot(range(T-1), Yt_DySAT_t_means[1][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1].plot(T, 3, color = 'white')
    axs[1].fill_between(range(T-1), Yt_DySAT_t_lowers[1][i], 
                          Yt_DySAT_t_uppers[1][i], color = colours[i], alpha=0.11)
axs[1].axvline(x=2.5, color='k', linestyle='--')
axs[1].axvline(x=4.5, color='k', linestyle='--')
axs[1].axvline(x=6.5, color='k', linestyle='--')
axs[1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1].set_title('d=32', fontsize = 11)

for i in range(K):
    axs[2].plot(range(T-1), Yt_DySAT_t_means[2][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[2].plot(T, 3, color = 'white')
    axs[2].fill_between(range(T-1), Yt_DySAT_t_lowers[2][i], 
                          Yt_DySAT_t_uppers[2][i], color = colours[i], alpha=0.11)
axs[2].axvline(x=2.5, color='k', linestyle='--')
axs[2].axvline(x=4.5, color='k', linestyle='--')
axs[2].axvline(x=6.5, color='k', linestyle='--')
axs[2].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[2].set_title('d=128', fontsize = 11)
# save the plot
plt.savefig('DySAT_dimensions.png')
# %%
