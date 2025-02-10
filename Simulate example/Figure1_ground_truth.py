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

Bs = np.array([B0,B1,B1, B0])
#%%
As = np.load('As.npy', allow_pickle=True)
As = As[[0,3,4, 7]]
Z = np.load('Z.npy', allow_pickle=True)
Cs = np.load('Cs.npy', allow_pickle=True)
Cs = Cs[[0,3,4, 7]]
# %%
T = len(Bs)
# %%
p = 150

mu1 = np.repeat(0, p)
mu1[20:75] = 1

mu2 = np.repeat(0, p)
mu2[80:140] = 1
mu3 = mu1
#%%
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
# do the same for Bcs
Bcs = [] 
for i in range(T):
    B = Bs[i]
    top = np.hstack([B, np.zeros((K, p))])
    if i  == 1:
        bottom = np.hstack([np.vstack([mu1,mu1,mu3]).T, np.zeros((p, p))])
    else:
        bottom = np.hstack([np.vstack([mu1,mu2,mu3]).T, np.zeros((p, p))])
    Bc = np.vstack([top, bottom])
    Bcs.append(Bc)

#%%
def SBM_dynamic_distbn(As, Bs, Z, d):
    return WSBM_dynamic_distbn(As, Bs, Bs*(1-Bs), Z, d)


def WSBM_dynamic_distbn(As, Bs, Cs, Z, d):
    T = As.shape[0]
    K = Bs.shape[1]
    A = np.block([A for A in As])
    B = np.block([B for B in Bs])
    P = np.block([B[Z,:][:,Z] for B in Bs])
    
    # Spectral embeddings
    UA, SA, VAt = np.linalg.svd(A); VA = VAt.T
    UB, SB, VBt = np.linalg.svd(B); VB = VBt.T
    UP, SP, VPt = np.linalg.svd(P); VP = VPt.T

    XB = UB[:,0:d] @ np.diag(np.sqrt(SB[0:d]))
    XP = UP[:,0:d] @ np.diag(np.sqrt(SP[0:d]))
    XZ = XB[Z,:]
        
    # Map spectral embeddings to latent positions
    print(UP.shape, UA.shape, VP.shape, VA.shape)
    UW, _, VWt = np.linalg.svd(UP[:,0:d].T @ UA[:,0:d] + VP[:,0:d].T @ VA[:,0:d])
    W = UW @ VWt
    L = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ XP
    R = np.linalg.inv(L.T) @ W


    Ys = np.zeros((T,K,d))
    Xs = XB @ L @ W
    for t in range(T):    
        Ys[t] = VB[t*K:(t+1)*K,0:d] @ np.diag(np.sqrt(SB[0:d])) @ R       
    
    return (Ys, Xs)

#%%
Acs = np.array(Acs).reshape(T, n+p, n+p)
Bcs = np.array(Bcs).reshape(T, K+p, K+p)
#%%
Zc = np.append(Z,list(range(3, 153)))
Yc, Xc = SBM_dynamic_distbn(Acs, Bs = Bcs, Z = Zc, d= 3)
# %%
# plot Yc
# plotting colors
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:K])
colours = [colours[0], colours[2], colours[1]]
# %%
fig, ax = plt.subplots(1,4, figsize=(20,5))
ax[0].scatter(Yc[0,:3,0], Yc[0,:3,1], c = colours, label = f'Time {i}', s=500)
ax[1].scatter(Yc[1,:3,0], Yc[1,:3,1], c = colours, label = f'Time {i}', s=500)
ax[2].scatter(Yc[2,:3,0], Yc[2,:3,1], c = colours, label = f'Time {i}', s=500)
ax[3].scatter(Yc[3,:3,0], Yc[3,:3,1], c = colours, label = f'Time {i}', s=500)
# make the axis be the same
for i in range(4):
    ax[i].set_xlim(-0.6,-0.2)
    ax[i].set_ylim(-0.4,0.4)
# %%
# repeat Yc 0 3 times 
# then Yc 1 2 times 
# then Yc 2 2 times
# then Yc 3 3 times
Yc = np.repeat(Yc, [3,2,2,3], axis = 0)

# %%
# plot the first dimension as a function of time
fig, ax = plt.subplots(1,1, figsize=(10,5))
for i in range(3):
    ax.plot(np.mean(Yc[:,i,:2], axis =1), "-o",label = f'Community {i}', c = colours[i], lw=3)
    # filp the Y axis
    ax.invert_yaxis()
    ax.axvline(x=2.5, color='k', linestyle='--')
    ax.axvline(x=4.5, color='k', linestyle='--')
    ax.axvline(x=6.5, color='k', linestyle='--')
# %%
# save Yc 
np.save('Yc.npy', Yc)