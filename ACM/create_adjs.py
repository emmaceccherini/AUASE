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
from ast import literal_eval
#%%
years = range(2000, 2015)
dfs = {}
for year in years:
    dfs[year] = pd.read_csv(f'dfs/df_{year}.csv')
# %%
# get the list of authors for each interval 
authors_list = {}
for interval, df in dfs.items():
    authors_list[interval] = df['Authors']
#%%
for year, lists in authors_list.items():
    authors_list[year] = [[name.strip() for name in literal_eval(authors)] for authors in lists]
#%%
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
        if isinstance(sublist, float):
            continue
        # ignore if the sublist has only one author
        if len(sublist) <1:
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
for interval in dfs.keys():
    print(interval)
    As.append(adj_matrix_from_authors_list(authors_list[interval]))
#%%
idx =[]
for i in range(n):
    sum  = np.sum([A[i,:].toarray() for A in As])
    if sum > 0:
        idx.append(i)
#%%
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
for i, A in enumerate(As):
    io.savemat(f'adjs/adj_matrix_{i}.mat', {'A': A})

# %%
# truncate all the elemnts of the adjacency matrices to 1
As_bin = []
for A in As:
        A_dense = A.toarray()  # convert to dense matrix
        As_bin.append(sparse.csr_matrix( np.minimum(A_dense, 1)))
# %%
for i, A in enumerate(As_bin):
    io.savemat(f'adjs/adj_matrix_{i}_bin.mat', {'A': A})

# %%
n = As_bin[0].shape[0]
for i in range(n):
    sum = np.sum([A[i,:].toarray() for A in As_bin])
    if sum <= 0:
        print(i)
# %%
