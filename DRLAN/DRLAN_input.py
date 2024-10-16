# %%
import numpy as np 
from scipy.sparse import csc_matrix
from scipy import sparse
import glob
from scipy import io
np.random.seed(0)
#%%
T = 9
# load As.npy and Z.npy
As = [io.loadmat(f'adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
# %%
# Get a list of all files that match the pattern
files = glob.glob('C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))
# %%
T = len(As)
n = As[0].shape[0]
# %%
Network = np.array([[csc_matrix(As[i]) for i in range(T)]], dtype=object)
# %%
Attributes = np.array([[csc_matrix(Cs[i]) for i in range(T)]], dtype=object)
# %%
data = {'Attributes': Attributes ,'Network': Network}
io.savemat('DBLP.mat', data)
# %%
