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
# load Z.npy
Z = np.load('Z.npy')
T = 10
n = 1000    
K = 3
# %%
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:K])
#%%
Z_new = np.zeros(n)
Z_new[Z==1] = int(2)
Z_new[Z==2] = int(1)
Z = Z_new
Z = Z.astype(int)
Zcol = colours[Z]
# %%
# load 
Yt_UASE = np.load('Yt_UASE.npy')
Yt_CUASE = np.load('Yt_CUASE.npy')
# %%
Yt_UASE_t = np.array(np.array_split(Yt_UASE, T)).reshape((T, n))
Yt_CUASE_t =np.array( np.array_split(Yt_CUASE, T)).reshape((T, n))
# %%
# compute the mean for each value of Z
Yt_UASE_t_mean = [np.mean(Yt_UASE_t[:,Z==i], axis = 1) for i in range(K)]
Yt_CUASE_t_mean = [np.mean(Yt_CUASE_t[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_UASE_t_lower = [np.percentile(Yt_UASE_t[:,Z==i], 5, axis =1) for i in range(K)]
Yt_UASE_t_upper = [np.percentile(Yt_UASE_t[:,Z==i], 95, axis =1) for i in range(K)]

Yt_CUASE_t_lower = [np.percentile(Yt_CUASE_t[:,Z==i], 5, axis =1) for i in range(K)]
Yt_CUASE_t_upper = [np.percentile(Yt_CUASE_t[:,Z==i], 95, axis =1) for i in range(K)]

# %%
Yt_DRLAN = np.load('Yt_DRLAN.npy')
# %%
Yt_DRLAN_t = np.array(np.array_split(Yt_DRLAN, T)).reshape((T, n))

# compute the mean for each value of Z
Yt_DRLAN_t_mean = [np.mean(Yt_DRLAN_t[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_DRLAN_t_lower = [np.percentile(Yt_DRLAN_t[:,Z==i], 5, axis =1) for i in range(K)]
Yt_DRLAN_t_upper = [np.percentile(Yt_DRLAN_t[:,Z==i], 95, axis =1) for i in range(K)]

# %%
Yt_DySAT = np.load('Yt_DySAT.npy')
# %%

# compute the mean for each value of Z
Yt_DySAT_t_mean = [np.mean(Yt_DySAT[:,Z==i], axis = 1) for i in range(K)]
# %%
Yt_DySAT_t_lower = [np.percentile(Yt_DySAT[:,Z==i], 5, axis =1) for i in range(K)]
Yt_DySAT_t_upper = [np.percentile(Yt_DySAT[:,Z==i], 95, axis =1) for i in range(K)]


#%%
# load 'Yc.npy'
Yc = np.load('Yc.npy')
coloursgt = [colours[0], colours[2], colours[1]]
Yc = Yc[:, :3, 0]
#%%
Yc[:3, 2] =0
Yc[:3, 1] =-1
Yc[:3, 0] =1
#%%
Yc[3:5, :] = 0
#%%
Yc[5:7, 0] = 0 
Yc[5:7, 2] = 0
Yc[5:7, 1] = -1
#%%
Yc[7:,0] = 1
Yc[7:, 1] = -1
Yc[7:, 2] = 0
#%%
#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])


# Ground truth plot in the center
ax_gt = fig.add_subplot(gs[0,0])


ax_gt.set_title('(a) Communities\' behaviours', fontsize = 12)
# ax_gt.set_xlim(0, 9)
# ax_ground_truth.set_yticks([])

ax_gt.scatter([0,1,2],[1,1,1], marker="o", c=coloursgt[0], s = 100)
ax_gt.scatter([0,1,2],[2,2,2], marker="*",c=coloursgt[2], s = 100)
ax_gt.scatter([0,1,2],[3,3,3], marker="s",c=coloursgt[1], s = 100)

ax_gt.scatter([3,4],[1,1], marker="*",c=coloursgt[0], s = 100)
ax_gt.scatter([3,4],[2,2],marker="*",c=coloursgt[2], s = 100)
ax_gt.scatter([3,4],[3,3], marker="*",c=coloursgt[1], s = 100)

ax_gt.scatter([5,6],[1,1], marker="*",c=coloursgt[0], s = 100)
ax_gt.scatter([5,6],[2,2], marker="*",c=coloursgt[2], s = 100)
ax_gt.scatter([5,6],[3,3], marker="s",c=coloursgt[1], s = 100)

ax_gt.scatter([7,8,9],[1,1,1], marker="o",c=coloursgt[0], s = 100)
ax_gt.scatter([7,8,9],[2,2,2], marker="*",c=coloursgt[2], s = 100)
ax_gt.scatter([7,8,9],[3,3,3], marker="s",c=coloursgt[1], s = 100)

ax_gt.invert_yaxis()
ax_gt.axvline(x=2.5, color='k', linestyle='--')
ax_gt.axvline(x=4.5, color='k', linestyle='--')
ax_gt.axvline(x=6.5, color='k', linestyle='--')
ax_gt.set_ylim(0.5, 3.5)
ax_gt.set_ylabel('Z', fontsize = 12)
# ax_gt.set_xlim(0, 9)
ax_gt.tick_params( labelsize = 12)

# Ground truth plot in the center
ax_ground_truth = fig.add_subplot(gs[0,1])

# ax_ground_truth.set_aspect('equal', adjustable="box")

  # Span across both columns
ax_ground_truth.set_title('(b) Noise-free embedding', fontsize = 12)
ax_ground_truth.set_xlim(0, 9)
# ax_ground_truth.set_yticks([])

for i in range(K):
    ax_ground_truth.plot(range(T),Yc[:,i], '-o',
                          c=coloursgt[i], lw=4, markersize = 7)
ax_ground_truth.invert_yaxis()
ax_ground_truth.axvline(x=2.5, color='k', linestyle='--')
ax_ground_truth.axvline(x=4.5, color='k', linestyle='--')
ax_ground_truth.axvline(x=6.5, color='k', linestyle='--')
# ax_ground_truth.set_ylim(-0.07, -0.32)
# ax_uase.set_xlim(0, 9)
ax_ground_truth.tick_params(left = False, labelleft = False, labelsize = 12)


# Plot on the second row
ax_uase = fig.add_subplot(gs[1, 0])
ax_uase.set_title('(c) UASE', fontsize = 12)
for i in range(K):
    ax_uase.plot(range(T), Yt_UASE_t_mean[i], 
                 '-o', c=colours[i], lw=3, markersize = 7)
    ax_uase.fill_between(range(T), Yt_UASE_t_lower[i], Yt_UASE_t_upper[i], color=colours[i], alpha=0.15)
ax_uase.axvline(x=2.5, color='k', linestyle='--')
ax_uase.axvline(x=4.5, color='k', linestyle='--')
ax_uase.axvline(x=6.5, color='k', linestyle='--')
# ax_uase.set_xlabel('Time')
ax_uase.set_ylabel('Embedding', fontsize = 12)
# ax_uase.set_xlim(0, 9)
ax_uase.tick_params(left = False, labelleft = False, labelsize = 12)
# flip the y axis
ax_uase.invert_yaxis()

ax_auase = fig.add_subplot(gs[1, 1])
ax_auase.set_title('(d) AUASE', fontsize = 12)
for i in range(K):
    ax_auase.plot(range(T), Yt_CUASE_t_mean[i], '-o',
                   c=colours[i], label=f'Community {i+1}', lw=3, markersize = 7)
    ax_auase.fill_between(range(T), Yt_CUASE_t_lower[i], Yt_CUASE_t_upper[i], color=colours[i], alpha=0.15)
ax_auase.axvline(x=2.5, color='k', linestyle='--')
ax_auase.axvline(x=4.5, color='k', linestyle='--')
ax_auase.axvline(x=6.5, color='k', linestyle='--')
# ax_auase.set_xlabel('Time')
# ax_auase.set_ylabel('Embedding')
# ax_auase.set_xlim(0, 9)
# flip the y axis
ax_auase.invert_yaxis()
ax_auase.tick_params(left = False, labelleft = False, labelsize = 12)

# Plot on the third row
ax_drlan = fig.add_subplot(gs[2, 0])
ax_drlan.set_title('(e) DRLAN',fontsize = 12)
for i in range(K):
    ax_drlan.plot(range(T), Yt_DRLAN_t_mean[i], '-o',
                   c=colours[i], lw=3, markersize = 7)
    ax_drlan.fill_between(range(T), Yt_DRLAN_t_lower[i], Yt_DRLAN_t_upper[i], color=colours[i], alpha=0.15)
ax_drlan.axvline(x=2.5, color='k', linestyle='--')
ax_drlan.axvline(x=4.5, color='k', linestyle='--')
ax_drlan.axvline(x=6.5, color='k', linestyle='--')
ax_drlan.set_xlabel('Time', fontsize = 12)
ax_drlan.set_ylabel('Embedding', fontsize = 12)
# ax_drlan.set_xlim(0, 9)
ax_drlan.tick_params(left = False, labelleft = False, labelsize = 12)
# flip the y axis
ax_drlan.invert_yaxis()

ax_dysat = fig.add_subplot(gs[2, 1])
ax_dysat.set_title('(f) DySAT', fontsize = 12)
for i in range(K):
    ax_dysat.plot(range(T-1), Yt_DySAT_t_mean[i], '-o', 
                  c=colours[i], lw=3, markersize = 7)
    ax_dysat.plot(T-1, 3, color = 'white')
    ax_dysat.fill_between(range(T-1), Yt_DySAT_t_lower[i], 
                          Yt_DySAT_t_upper[i], color=colours[i], alpha=0.15)
ax_dysat.axvline(x=2.5, color='k', linestyle='--')
ax_dysat.axvline(x=4.5, color='k', linestyle='--')
ax_dysat.axvline(x=6.5, color='k', linestyle='--')
ax_dysat.set_xlabel('Time', fontsize = 12)
# make teh font of the ticks bigger
# ax_dysat.set_xticks(fontsize = 11)

#set the x limit
# ax_dysat.set_xlim(-0.4, 9.4)
ax_dysat.tick_params(left = False, labelleft = False, labelsize = 12)

# Hide the unused subplot in the third row
axes_empty = fig.add_subplot(gs[2, 1])
axes_empty.axis('off')

bev1 = ax_gt.scatter([],[], marker="o", c= "grey", s = 100, label = 'Behaviour 1')
bev2 = ax_gt.scatter([],[], marker="*",c="grey", s = 100, label = 'Behaviour 2')
bev3 = ax_gt.scatter([],[], marker="s",c="grey", s = 100, label = 'Behaviour 3')
comm1 = ax_gt.plot([],[], '-o',
                   c=colours[0], lw=3, markersize = 7, 
                   label = 'Community 1')
comm2 = ax_gt.plot([],[], '-o',
                   c=colours[1], lw=3, markersize = 7, 
                   label = 'Community 2')
comm3 = ax_gt.plot([],[], '-o',
                   c=colours[2], lw=3, markersize = 7, 
                   label = 'Community 3')
# Add the legend to the figure
legend = fig.legend(handles = [comm1[0],  bev1, comm2[0], bev2,comm3[0], bev3], 
    loc='lower center', ncol=K,
           prop={'size': 12})   

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.09, 1, 0.97])

plt.savefig('Figure1_new.png', dpi = 300)

