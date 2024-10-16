# %%
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
# %%
d = 3
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
Yt_CUASE_t_means = []
Yt_CUASE_t_lowers = []
Yt_CUASE_t_uppers = []
for t,alpha in enumerate(alphas):
    Acs = []
    for i in range(T):
        A = As[i]
        C = Cs[i]
        top = np.hstack([(1-alpha)*A, alpha * C])
        bottom = np.hstack([alpha* C.T, np.zeros((p, p))])
        Ac = np.vstack([top, bottom])
        Acs.append(Ac)

    _, YAs_CUASE = se.UASE(Acs,d)
    YAs_CUASE = YAs_CUASE[:,:n,:]

    YAs_CUASE = YAs_CUASE.reshape((T*n, K))

    reducer = umap.UMAP(n_components=1, random_state=0)

    Yt_CUASE = reducer.fit_transform(YAs_CUASE)

    Yt_CUASE_t =np.array( np.array_split(Yt_CUASE, T)).reshape((T, n))
    # compute the mean for each value of Z
    Yt_CUASE_t_means.append([np.mean(Yt_CUASE_t[:,Z==i], axis = 1) for i in range(K)])
    Yt_CUASE_t_lowers.append([np.percentile(Yt_CUASE_t[:,Z==i], 5, axis =1) for i in range(K)])
    Yt_CUASE_t_uppers.append([np.percentile(Yt_CUASE_t[:,Z==i], 95, axis =1) for i in range(K)])
    
# %%
# plot the results
fig, ax = plt.subplots(3,3, figsize = (15,15))
for i in range(9):
    l = i//3
    s = i%3
    for k in range(K):
        ax[l,s].plot(Yt_CUASE_t_means[i][k], label = f'Z = {k}', color = colours[k])
        ax[l,s].fill_between(range(10), Yt_CUASE_t_lowers[i][k], Yt_CUASE_t_uppers[i][k], color = colours[k], alpha = 0.3)
    ax[l,s].set_title(f'alpha = {alphas[i]}') 
# save the figure
plt.savefig('CUASE_compare_alpha.png')      
# %%
ds = [16,32,64]
Yt_CUASE_t_means = []
Yt_CUASE_t_lowers = []
Yt_CUASE_t_uppers = []
alpha =0.2
for t,d in enumerate(ds):
    Acs = []
    for i in range(T):
        A = As[i]
        C = Cs[i]
        top = np.hstack([(1-alpha)*A, alpha * C])
        bottom = np.hstack([alpha* C.T, np.zeros((p, p))])
        Ac = np.vstack([top, bottom])
        Acs.append(Ac)

    _, YAs_CUASE = se.UASE(Acs,d)
    YAs_CUASE = YAs_CUASE[:,:n,:]

    YAs_CUASE = YAs_CUASE.reshape((T*n, d))

    reducer = umap.UMAP(n_components=1, random_state=0)

    Yt_CUASE = reducer.fit_transform(YAs_CUASE)

    Yt_CUASE_t =np.array( np.array_split(Yt_CUASE, T)).reshape((T, n))
    # compute the mean for each value of Z
    Yt_CUASE_t_means.append([np.mean(Yt_CUASE_t[:,Z==i], axis = 1) for i in range(K)])
    Yt_CUASE_t_lowers.append([np.percentile(Yt_CUASE_t[:,Z==i], 5, axis =1) for i in range(K)])
    Yt_CUASE_t_uppers.append([np.percentile(Yt_CUASE_t[:,Z==i], 95, axis =1) for i in range(K)])
    
# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 20))

for i in range(K):
    axs[0].plot(range(T), Yt_CUASE_t_means[0][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0].plot(T, 3, color = 'white')
    axs[0].fill_between(range(T), Yt_CUASE_t_lowers[0][i], 
                          Yt_CUASE_t_uppers[0][i], color = colours[i], alpha=0.11)
axs[0].axvline(x=2.5, color='k', linestyle='--')
axs[0].axvline(x=4.5, color='k', linestyle='--')
axs[0].axvline(x=6.5, color='k', linestyle='--')
axs[0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0].set_title('d=16', fontsize = 11)

for i in range(K):
    axs[1].plot(range(T), Yt_CUASE_t_means[1][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1].plot(T, 3, color = 'white')
    axs[1].fill_between(range(T), Yt_CUASE_t_lowers[1][i], 
                          Yt_CUASE_t_uppers[1][i], color = colours[i], alpha=0.11)
axs[1].axvline(x=2.5, color='k', linestyle='--')
axs[1].axvline(x=4.5, color='k', linestyle='--')
axs[1].axvline(x=6.5, color='k', linestyle='--')
axs[1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1].set_title('d=32', fontsize = 11)

for i in range(K):
    axs[2].plot(range(T), Yt_CUASE_t_means[2][i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[2].plot(T, 3, color = 'white')
    axs[2].fill_between(range(T), Yt_CUASE_t_lowers[2][i], 
                          Yt_CUASE_t_uppers[2][i], color = colours[i], alpha=0.11)
axs[2].axvline(x=2.5, color='k', linestyle='--')
axs[2].axvline(x=4.5, color='k', linestyle='--')
axs[2].axvline(x=6.5, color='k', linestyle='--')
axs[2].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[2].set_title('d=128', fontsize = 11)
# save the plot
plt.savefig('AUASE_dimensions.png')
# %%
