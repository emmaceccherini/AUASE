#%% 
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import numba as nb
from scipy import io
import glob
import pickle
import ast
import scipy.sparse as sparse
#%%
# Find all CSV files in the current directory that start with 'df'
csv_files = glob.glob('dfs/df*.csv')

# Load each file into a DataFrame
dfs = {file: pd.read_csv(file) for file in csv_files}
#%%
# change the keys to the interval
new_dfs = {}
for key in list(dfs.keys()):
    interval = key.split('_')[-1].split('.')[0]
    new_dfs[interval] = dfs[key]
dfs = new_dfs

def extract_first_year(key):
    return int(key.split(' - ')[0])

# Sort the keys and reorder the dictionary
sorted_keys = sorted(dfs.keys(), key=extract_first_year)
dfs = {key: dfs[key] for key in sorted_keys}
# %%
# get the list of authors for each interval 
authors_list = {}
for interval, df in dfs.items():
    authors_list[interval] = df['Authors']

# %% 	
# load active authors
with open('active_authors.pkl', 'rb') as f:
    unique_authors = pickle.load(f)
n = len(unique_authors)


#%%
# Create a dictionary to map author names to indices
author_indices = {author: index for index, author in enumerate(unique_authors)}

#%%
# Count collaborations
nb.jit()
def adj_matrix_from_authors_list(authors_list):
    adj_matrix = lil_matrix((n, n), dtype=np.int32)
    for sublist in authors_list:
        # ignore is the sublist is nan
        # print(sublist)
        sublist = ast.literal_eval(sublist)
        if isinstance(sublist, float):
            continue
        # ignore if the sublist has only one author
        if len(sublist) <1:
            continue
        # check that at least two of the authors are active
        active_authors_in_sublist = [author for author in sublist if author in author_indices]
        if len(active_authors_in_sublist) < 2:
            continue
        for i in range(len(sublist)):
        # find the indeces of the authors
            if sublist[i] not in author_indices:
                continue
            author1 = author_indices[sublist[i]]
            for j in range(i+1, len(sublist)):
                if sublist[j] not in author_indices:
                    continue
                author2 = author_indices[sublist[j]]
                adj_matrix[author1, author2] += 1
                adj_matrix[author2, author1] += 1
    return adj_matrix
#%%
As = []
idx_remove = []
for interval in dfs.keys():
    print(interval)	
    As.append(adj_matrix_from_authors_list(authors_list[interval]))
# %%
# remove from author_indices the authors that are not active
idx =[]
for i in range(n):
    sum  = np.sum([A[i,:].toarray() for A in As])
    if sum > 0:
        idx.append(i)
# %%
idx = np.array(idx)
#%%
#remove from the adjacency matrices the authors that are not active
As = [A[idx,:][:,idx] for A in As]
#%%
# update the author_indices
author_indices = {author: index for index, author in enumerate([unique_authors[i] for i in idx])}
# %%
# save the author_indices
with open('author_indices.pkl', 'wb') as f:
    pickle.dump(author_indices, f)
# %%
#update the active authors
active_authors = [unique_authors[i] for i in idx]
# %%
# save the active authors
with open('active_authors.pkl', 'wb') as f:
    pickle.dump(active_authors, f)

#%%
# save the adjacency matrices
for i, A in enumerate(As):
    io.savemat(f'adj/adj_matrix_{i}.mat', {'A': A})
# %%
# truncate all the elemnts of the adjacency matrices to 1
As_bin = []
for A in As:
        A_dense = A.toarray()  # convert to dense matrix
        As_bin.append(sparse.csr_matrix( np.minimum(A_dense, 1)))
# %%
for i, A in enumerate(As_bin):
    io.savemat(f'adj/adj_matrix_{i}_bin.mat', {'A': A})


# %%
# sanity check 
for i in range(As[0].shape[0]):
    sum = np.sum([A[i,:].toarray() for A in As])
    if sum <= 0:
        print(i)
# %%
