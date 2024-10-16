#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import umap
#%%
# load Z.npy
Z = np.load('../Z.npy')
#%%
# GloDyNe embeddings for d= 3, 16, 32, 64 in the same directory
#%%
T = 10
n = 1000
K = 3
reducer = umap.UMAP(n_components=1, random_state=0)
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
embeddings = np.load('fig1_16.npy', allow_pickle = True)    
Y_Gd= [] 
for i in range(T):
    Y_Gd.append(np.vstack(list(embeddings[i].values())))
#%%
Yt_Gd= np.zeros((T, n))
for t in range(T):
    Yt_Gd[t] = reducer.fit_transform(Y_Gd[t]).reshape(n)

# %%
Yt_Gd_mean_16 = [np.mean(Yt_Gd[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_Gd_lower_16 = [np.percentile(Yt_Gd[:,Z==i], 5, axis =1) for i in range(K)]
Yt_Gd_upper_16 = [np.percentile(Yt_Gd[:,Z==i], 95, axis =1) for i in range(K)]
#%%
embeddings = np.load('fig1_32.npy', allow_pickle = True)    
Y_Gd= [] 
for i in range(T):
    Y_Gd.append(np.vstack(list(embeddings[i].values())))
#%%
Yt_Gd= np.zeros((T, n))
for t in range(T):
    Yt_Gd[t] = reducer.fit_transform(Y_Gd[t]).reshape(n)

# %%
Yt_Gd_mean_32 = [np.mean(Yt_Gd[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_Gd_lower_32 = [np.percentile(Yt_Gd[:,Z==i], 5, axis =1) for i in range(K)]
Yt_Gd_upper_32 = [np.percentile(Yt_Gd[:,Z==i], 95, axis =1) for i in range(K)]
#%%
embeddings = np.load('fig1_64.npy', allow_pickle = True)    
Y_Gd= [] 
for i in range(T):
    Y_Gd.append(np.vstack(list(embeddings[i].values())))
#%%
Yt_Gd= np.zeros((T, n))
for t in range(T):
    Yt_Gd[t] = reducer.fit_transform(Y_Gd[t]).reshape(n)

# %%
Yt_Gd_mean_64 = [np.mean(Yt_Gd[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_Gd_lower_64 = [np.percentile(Yt_Gd[:,Z==i], 5, axis =1) for i in range(K)]
Yt_Gd_upper_64 = [np.percentile(Yt_Gd[:,Z==i], 95, axis =1) for i in range(K)]
#%%
embeddings = np.load('fig1_3.npy', allow_pickle = True)    
Y_Gd= [] 
for i in range(T):
    Y_Gd.append(np.vstack(list(embeddings[i].values())))
#%%
Yt_Gd= np.zeros((T, n))
for t in range(T):
    Yt_Gd[t] = reducer.fit_transform(Y_Gd[t]).reshape(n)

# %%
Yt_Gd_mean_3 = [np.mean(Yt_Gd[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_Gd_lower_3 = [np.percentile(Yt_Gd[:,Z==i], 5, axis =1) for i in range(K)]
Yt_Gd_upper_3 = [np.percentile(Yt_Gd[:,Z==i], 95, axis =1) for i in range(K)]

#%%
# plot the 4 plots in one figure 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(K):
    axs[0,0].plot(range(T), Yt_Gd_mean_3[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0,0].plot(T, 3, color = 'white')
    axs[0,0].fill_between(range(T), Yt_Gd_lower_3[i], 
                          Yt_Gd_upper_3[i], color=colours[i], alpha=0.11)
axs[0,0].axvline(x=2.5, color='k', linestyle='--')
axs[0,0].axvline(x=4.5, color='k', linestyle='--')
axs[0,0].axvline(x=6.5, color='k', linestyle='--')
axs[0,0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0,0].set_title('d=3', fontsize = 11)

for i in range(K):
    axs[0,1].plot(range(T), Yt_Gd_mean_16[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[0,1].plot(T, 3, color = 'white')
    axs[0,1].fill_between(range(T), Yt_Gd_lower_3[i], 
                          Yt_Gd_upper_3[i], color=colours[i], alpha=0.11)
axs[0,1].axvline(x=2.5, color='k', linestyle='--')
axs[0,1].axvline(x=4.5, color='k', linestyle='--')
axs[0,1].axvline(x=6.5, color='k', linestyle='--')
axs[0,1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[0,1].set_title('d=16', fontsize = 11)

for i in range(K):
    axs[1,0].plot(range(T), Yt_Gd_mean_32[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1,0].plot(T, 3, color = 'white')
    axs[1,0].fill_between(range(T), Yt_Gd_lower_3[i], 
                          Yt_Gd_upper_3[i], color=colours[i], alpha=0.11)
axs[1,0].axvline(x=2.5, color='k', linestyle='--')
axs[1,0].axvline(x=4.5, color='k', linestyle='--')
axs[1,0].axvline(x=6.5, color='k', linestyle='--')
axs[1,0].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1,0].set_title('d=32', fontsize = 11)

for i in range(K):
    axs[1,1].plot(range(T), Yt_Gd_mean_64[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    axs[1,1].plot(T, 3, color = 'white')
    axs[1,1].fill_between(range(T), Yt_Gd_lower_3[i], 
                          Yt_Gd_upper_3[i], color=colours[i], alpha=0.11)
axs[1,1].axvline(x=2.5, color='k', linestyle='--')
axs[1,1].axvline(x=4.5, color='k', linestyle='--')
axs[1,1].axvline(x=6.5, color='k', linestyle='--')
axs[1,1].set_xlabel('Time', fontsize = 11)
# add a title with the number of dimensions
axs[1,1].set_title('d=64', fontsize = 11)
# save the plot
plt.savefig('fig1_compare_glodyne.png')
# %%
