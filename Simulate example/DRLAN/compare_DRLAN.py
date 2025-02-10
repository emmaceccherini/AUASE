# %%
import numpy as np
import matplotlib.pyplot as plt
import spectral_embedding as se
import matplotlib as mpl
import os
import scipy.io as sio
np.random.seed(123)
import umap
#%%
# load Z.npy
Z = np.load('../Z.npy')
#%%
# DRLAN embeddings for d= 3, 16, 32, 64  and various option and beta in the same directory
#%%
T = 10
n = 1000
K = 3
# for DRLAN my alpha =0.3 is equivalent to beta = 0.7 
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
def get_DRLAN_embs(beta,option,dim):
    data_DRLAN = []
    for i in range(1,T+1):
        # load embedding_i_03.mat
        data_DRLAN.append(sio.loadmat(f'embedding_{i}beta{beta}_option{option}_d{dim}.mat'))
    # vstack the embeddings
    Y_DRLAN = np.vstack([data_DRLAN[i]["U"] for i in range(T)])
    # do TSNE from 128 to 1 dimensions
    reducer = umap.UMAP(n_components=1, random_state=0)
    Yt_DRLAN = reducer.fit_transform(Y_DRLAN)
    Yt_DRLAN_t = np.array(np.array_split(Yt_DRLAN, T)).reshape((T, n))

    # compute the mean for each value of Z
    Yt_DRLAN_t_mean = [np.mean(Yt_DRLAN_t[:,Z==i], axis = 1) for i in range(K)]
    Yt_DRLAN_t_lower = [np.percentile(Yt_DRLAN_t[:,Z==i], 5, axis =1) for i in range(K)]
    Yt_DRLAN_t_upper = [np.percentile(Yt_DRLAN_t[:,Z==i], 95, axis =1) for i in range(K)]
    
    return Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper

#%%
beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.95,0.98]

Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,1,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 1 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color = colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # make the last plot empty - remove the axis
fig.delaxes(axes[2, 3])

    # axes[l,c].legend()
# 0.9 is the one that looks the best 
# save the plot
plt.savefig('DRLAN_option1.png')
# %%
Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,2,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 2 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color = colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # axes[l,c].legend()
fig.delaxes(axes[2, 3])
# 0.5 0r 0.7 looks the best
plt.savefig('DRLAN_option2.png') 
# %%
# %%
Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,3,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 3 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color = colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # axes[l,c].legend()
fig.delaxes(axes[2, 3])
# 0.7 looks the best 
plt.savefig('DRLAN_option3.png')
# %%
Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,4,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 4 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color = colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # axes[l,c].legend()
fig.delaxes(axes[2, 3])
# none of these look good  
plt.savefig('DRLAN_option4.png')
# %%
Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,5,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 5 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color = colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # axes[l,c].legend()
fig.delaxes(axes[2, 3])
# none of these look goods. 
plt.savefig('DRLAN_option5.png')
# %%
Yt_DRLAN_t_means = []
Yt_DRLAN_t_lowers = []
Yt_DRLAN_t_uppers = []
for t,b in enumerate(beta):
    Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper = get_DRLAN_embs(b,6,3)
    Yt_DRLAN_t_means.append(Yt_DRLAN_t_mean)
    Yt_DRLAN_t_lowers.append(Yt_DRLAN_t_lower)
    Yt_DRLAN_t_uppers.append(Yt_DRLAN_t_upper)

# %%
# option 6 
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
for t in range(len(beta)):
    l = t//4
    c = t%4
    for i in range(K):
        axes[l,c].plot(Yt_DRLAN_t_means[t][i], label = f'Z={i}', color = colours[i])
        axes[l,c].fill_between(range(T), Yt_DRLAN_t_lowers[t][i], Yt_DRLAN_t_uppers[t][i], color =colours[i], alpha = 0.3)
    axes[l,c].set_title(f'beta = {beta[t]}')
    # axes[l,c].legend()
fig.delaxes(axes[2, 3])
# none of these are good
plt.savefig('DRLAN_option6.png')
#%%
# compare the dimension 
def get_DRLAN_embs(beta,dim):
    data_DRLAN = []
    for i in range(1,T+1):
        # load embedding_i_03.mat
        data_DRLAN.append(sio.loadmat(f'embedding_{i}beta{beta}_d{dim}.mat'))
    # vstack the embeddings
    Y_DRLAN = np.vstack([data_DRLAN[i]["U"] for i in range(T)])
    # do TSNE from 128 to 1 dimensions
    reducer = umap.UMAP(n_components=1, random_state=0)
    Yt_DRLAN = reducer.fit_transform(Y_DRLAN)
    Yt_DRLAN_t = np.array(np.array_split(Yt_DRLAN, T)).reshape((T, n))

    # compute the mean for each value of Z
    Yt_DRLAN_t_mean = [np.mean(Yt_DRLAN_t[:,Z==i], axis = 1) for i in range(K)]
    Yt_DRLAN_t_lower = [np.percentile(Yt_DRLAN_t[:,Z==i], 5, axis =1) for i in range(K)]
    Yt_DRLAN_t_upper = [np.percentile(Yt_DRLAN_t[:,Z==i], 95, axis =1) for i in range(K)]
    
    return Yt_DRLAN_t_mean, Yt_DRLAN_t_lower, Yt_DRLAN_t_upper

#%%
Yt_DRLAN_t_mean_16, Yt_DRLAN_t_lower_16, Yt_DRLAN_t_upper_16 = get_DRLAN_embs(0.7,16)
#%%
# d = 32
Yt_DRLAN_t_mean_32, Yt_DRLAN_t_lower_32, Yt_DRLAN_t_upper_32 = get_DRLAN_embs(0.7,32)
#%%
# d = 64
Yt_DRLAN_t_mean_64, Yt_DRLAN_t_lower_64, Yt_DRLAN_t_upper_64 = get_DRLAN_embs(0.7,64)
#%%
# plot the 4 plots in one figure 2x2
fig, axs = plt.subplots(3, 1, figsize=(10, 20))

for i in range(K):
    axs[0].plot(range(T), Yt_DRLAN_t_mean_16[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0].plot(T, 3, color = 'white')
    axs[0].fill_between(range(T), Yt_DRLAN_t_lower_16[i], 
                          Yt_DRLAN_t_upper_16[i], color = colours[i], alpha=0.11)
axs[0].axvline(x=2.5, color='k', linestyle='--')
axs[0].axvline(x=4.5, color='k', linestyle='--')
axs[0].axvline(x=6.5, color='k', linestyle='--')
axs[0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0].set_title('d=16', fontsize = 11)

for i in range(K):
    axs[1].plot(range(T), Yt_DRLAN_t_mean_32[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1].plot(T, 3, color = 'white')
    axs[1].fill_between(range(T), Yt_DRLAN_t_lower_32[i], 
                          Yt_DRLAN_t_upper_32[i], color = colours[i], alpha=0.11)
axs[1].axvline(x=2.5, color='k', linestyle='--')
axs[1].axvline(x=4.5, color='k', linestyle='--')
axs[1].axvline(x=6.5, color='k', linestyle='--')
axs[1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1].set_title('d=32', fontsize = 11)

for i in range(K):
    axs[2].plot(range(T), Yt_DRLAN_t_mean_64[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[2].plot(T, 3, color = 'white')
    axs[2].fill_between(range(T), Yt_DRLAN_t_lower_64[i], 
                          Yt_DRLAN_t_upper_64[i], color = colours[i], alpha=0.11)
axs[2].axvline(x=2.5, color='k', linestyle='--')
axs[2].axvline(x=4.5, color='k', linestyle='--')
axs[2].axvline(x=6.5, color='k', linestyle='--')
axs[2].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[2].set_title('d=128', fontsize = 11)
# save the plot
plt.savefig('DRLAN_dimensions.png')
# %%
