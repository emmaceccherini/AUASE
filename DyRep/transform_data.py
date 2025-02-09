#%%
# Load adjacency matrices from ACM dataset and filter out inactive nodes

import scipy.io
import numpy as np
import pandas as pd
#%%

As = []
for i in range(11):
    A= scipy.io.loadmat(f'adj_epinions/adj_matrix_{i}.mat')["A"]
    # A = A.todense()
    As.append(A)
#%%
# 2. Compute the sum over all adjacency matrices for each node
# Assuming A is a sparse matrix. If A is dense, adjust accordingly.
sum_adj = np.zeros(As[0].shape[0])

for A in As:
    if isinstance(A, (np.matrix, np.ndarray)):
        sum_adj += A.sum(axis=1).A1 if hasattr(A.sum(axis=1), 'A1') else A.sum(axis=1)
    else:
        # Handle other types if necessary
        sum_adj += A.sum(axis=1).A1

# 3. Identify active nodes (sum >0)
active_nodes = np.where(sum_adj > 0)[0]
print(f"Total nodes in original adjacency matrices: {As[0].shape[0]}")
print(f"Number of active nodes (with at least one interaction): {len(active_nodes)}")
#%%
# 4. Create a mapping from original node IDs to new node IDs for active nodes
# This ensures that node IDs are contiguous and start from 0
node_id_mapping = {original_id: new_id for new_id, original_id in enumerate(active_nodes)}
print(f"Node ID mapping created for {len(node_id_mapping)} active nodes.")

# 5. Process adjacency matrices to include only active nodes
As_active = []
for i, A in enumerate(As):
    # Select only active nodes for both rows and columns
    A_active = A[active_nodes, :][:, active_nodes]
    As_active.append(A_active)
    print(f"Processed adjacency matrix {i} shape: {A_active.shape}")
#%%

# Assuming `As` is a list of sparse matrices and `timestamps` is defined
data = []

for t_idx, matrix in enumerate(As_active):  # Iterate through adjacency matrices
    timestamp = t_idx
    rows, cols = matrix.nonzero()  # Get non-zero entries (u, i pairs)
    print(timestamp)
    # Add each interaction to the data list
    for u, i in zip(rows, cols):
        data.append({
            "event_types": 0,  # Assuming event_types is a constant
            "u": u,
            "i": i,
            "ts": int(timestamp)  # Convert timestamp to integer or epoch format
        })

# Create the DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV for inspection (optional)
df.to_csv("Epinions.csv", index=False)


