#%%
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import scipy.io as scio
import glob 
#%%
# Find all the files that start with 'labels' and end with '.pkl'
files = glob.glob('data/labels/labels*.pkl')

# Load each file
labels = {}
for file in files:
    with open(file, 'rb') as f:
        labels[file] = pkl.load(f)
#%%
new_labels = {}
for key in list(labels.keys()):
    interval = key.split('_')[-1].split('.')[0]
    new_labels[interval] = labels[key]
labels = new_labels

def extract_first_year(key):
    return int(key.split(' - ')[0])

# Sort the keys and reorder the dictionary
sorted_keys = sorted(labels.keys(), key=extract_first_year)
labels = {key: labels[key] for key in sorted_keys}
#%%
from sklearn.preprocessing import LabelEncoder
labels_encoded  ={}
le = LabelEncoder()
for key, values in labels.items():
    labels_encoded[key] = le.fit_transform(labels[key])
#%%
#%%
dataset_str = "DBLP"
for i in range(10):
    # load adj.mat and word_count_matrix_1996 - 2000.npz
    adj = scio.loadmat(f'data/adj/adj_matrix_{i}_bin.mat')["A"]
    # intervals = ["2000","2001","2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014"]
    intervals = list(labels_encoded.keys())
    # print(intervals[i])
    word_count_matrix = sp.load_npz(f'data/C_matrices/word_count_matrix_{intervals[i]}.npz')
    # CHANGE THE YEAR IN THE FILE NAME
    n = adj.shape[0]
    
    perm = np.random.permutation(n)
    train_idx = perm[:n-1].astype(int)
    test_idx = perm[n-1:]
    x = word_count_matrix
    y = np.array(labels_encoded[intervals[i]]).reshape(-1, 1)
    tx = word_count_matrix[train_idx]
    ty = np.array(labels_encoded[intervals[i]])[train_idx].reshape(-1, 1)
    allx = word_count_matrix[test_idx]
    ally = np.array(labels_encoded[intervals[i]])[test_idx].reshape(-1, 1)
    test_indices = test_idx


    graph = {i: list(np.where(row)[0]) for i, row in enumerate(adj.toarray())}

    # Convert to sparse matrices
    x_sparse = sp.csr_matrix(x)
    tx_sparse = sp.csr_matrix(tx)
    allx_sparse = sp.csr_matrix(allx)

    with open(f"data/ind.{dataset_str}_{i}.x", 'wb') as f:
        pkl.dump(x_sparse, f)

    with open(f"data/ind.{dataset_str}_{i}.y", 'wb') as f:
       pkl.dump(y, f)

    with open(f"data/ind.{dataset_str}_{i}.tx", 'wb') as f:
        pkl.dump(tx_sparse, f)

    with open(f"data/ind.{dataset_str}_{i}.ty", 'wb') as f:
       pkl.dump(ty, f)

    with open(f"data/ind.{dataset_str}_{i}.allx", 'wb') as f:
        pkl.dump(allx_sparse, f)

    with open(f"data/ind.{dataset_str}_{i}.ally", 'wb') as f:
       pkl.dump(ally, f)

    with open(f"data/ind.{dataset_str}_{i}.graph", 'wb') as f:
        pkl.dump(graph, f)

    with open(f"data/ind.{dataset_str}_{i}.test.index", 'w') as f:
        for index in test_indices:
            f.write(f"{index}\n")
    print(f"Saved {i}")

