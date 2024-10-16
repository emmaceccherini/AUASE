#%%
import numpy as np
import scipy.io as io
from screenot.ScreeNOT import *
from scipy import sparse
from sklearn.preprocessing import normalize
import glob
#%%
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(9)]
# Get a list of all files that match the pattern
files = glob.glob('../C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))
#%%
alpha = 0.7
Acs = []
for i in range(len(As)):
    A = As[i]
    C = Cs[i]
    p = Cs[0].shape[1]
    # standardize the columns of C
    C = normalize(C, axis=0)
    top = sparse.hstack([(1-alpha)*A, alpha * C])
    bottom = sparse.hstack([alpha * C.T, sparse.csr_matrix((p, p))])
    Ac = sparse.vstack([top, bottom])
    Acs.append(Ac)
#%%
def ScreeNOT_sparse(Y, k, strategy='i'):
    U, fY, Vt = sparse.linalg.svds(Y, k=2*k+2)
    gamma = np.min( [ Y.shape[0]/Y.shape[1], Y.shape[1]/Y.shape[0] ] )
    fZ = createPseudoNoise(fY, k, strategy=strategy)
    Topt = computeOptThreshold(fZ, gamma)
    fY_new = fY*(fY>Topt)
    # Xest = U @ np.diag(fY_new) @ Vt
    Xest = sparse.csr_matrix(U) @ sparse.csr_matrix(np.diag(fY_new)) @ sparse.csr_matrix(Vt)
    r = np.sum(fY_new>0)
    return Xest, Topt, r
#%%
_, _, dAC = ScreeNOT_sparse(sparse.hstack(Acs), 150) 

# %%
print(dAC)
#%%
_, _, dAC = ScreeNOT_sparse(sparse.hstack(As), 150) 

print(dAC)
# %%