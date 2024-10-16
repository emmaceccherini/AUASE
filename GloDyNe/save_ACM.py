#%%
from src.libne.DynWalks import DynWalks
import networkx as nx
import numpy as np
import scipy.io as io
from scipy.sparse import csc_matrix
import time 
As = [io.loadmat(f'adjs_ACM/adj_matrix_{i}_bin.mat')['A'].astype(float) for i in range(15)]
d = 128
T = len(As)
n = As[0].shape[0]
#%%
As_combined = []
As_combined.append(As[0])
for i in range(1, T):
    As_new = As[i] + As_combined[i-1]
    # make it bynary
    As_new[As_new > 0] = 1
    As_combined.append(As_new)
#%%%
G = []
# To track which nodes have been introduced so far
seen_nodes = set()

# Get the initial set of non-zero nodes from the first time step
idx = np.where(np.sum(As_combined[0], axis=1) != 0)[0]
seen_nodes.update(idx)  # Add these nodes to the seen set

# Create the graph for time step 0
G0 = As_combined[0][idx,:][:, idx]
G.append(csc_matrix(G0))
#%%
# Iterate over the subsequent time steps
for t in range(1, T):
    # Find the nodes that have non-zero connections in the current time step
    current_nonzero_idx = np.where(np.sum(As_combined[t], axis=1) != 0)[0]
    
    # Update the seen nodes to include any new nodes introduced at this time step
    seen_nodes.update(current_nonzero_idx)

    # Create a list of indices corresponding to all the seen nodes up to this point
    seen_nodes_list = np.array(sorted(seen_nodes))

    # Convert As_combined[t] to a sparse matrix to save memory
    As_t_sparse = As_combined[t]

    # Create a submatrix with all the seen nodes so far using sparse indexing
    Gt = As_t_sparse[np.ix_(seen_nodes_list, seen_nodes_list)]
    
    # Append the sparse matrix
    G.append(Gt)

#%%
graphs = []
for t in range(T):

    graphs.append(nx.from_scipy_sparse_matrix(G[t]))
print("data loaded")
print(d)
start = time.time()
model = DynWalks(
    G_dynamic=graphs,
    limit=0.1,
    num_walks=10,
    walk_length=80,
    window=10,
    emb_dim=d,
    negative=5,
    workers=32,
    seed=2019,
    scheme=4,
)
ya_raw = model.sampling_traning()
end_time = time.time()  
print("Time: ", end_time - start)
# save embeddings
np.save('embeddings/ACM.npy', ya_raw)

# %%
