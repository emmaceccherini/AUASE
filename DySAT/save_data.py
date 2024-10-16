
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import io
from scipy import sparse
import glob
# load data
As = [io.loadmat('adjs/adj_matrix_{}_bin.mat'.format(i))['A'].astype(np.float64) for i in range(15)]	

#As = np.array([A.todense() for A in As])
T = len(As)

def from_A_to_G(A):
    A =A.todense()
    G = nx.MultiGraph()

    num_nodes = A.shape[0]
    G.add_nodes_from(range(num_nodes))
    # Iterate through the NumPy array to add edges
    rows, cols = np.where(A > 0)

    for row, col in zip(rows, cols):
        G.add_edge(row, col)
    return G
        

graphs = np.empty(T, dtype=object)
graphs[:] = [from_A_to_G(A) for A in As]

# save it as a .npz file
np.savez('graphs.npz', graph=graphs, allow_pickle=True, encoding='latin1', fix_imports=True)


files = glob.glob('C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))

feats = np.empty(T, dtype=object)
feats[:] = [sp.csr_matrix(C,dtype=float) for C in Cs]

# save it as a .npz file
np.savez('features.npz', feats=feats, allow_pickle=True, encoding='latin1', fix_imports=True)


