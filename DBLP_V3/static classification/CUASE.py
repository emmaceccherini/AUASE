#%%
import numpy as np
import spectral_embedding as se
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from scipy import io
import glob
#%%
# load adj_matrix_0.mat to adj_matrix_11.mat
#%% CHECK SHOULD BE 10 
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(9)]

#%%
# Get a list of all files that match the pattern
files = glob.glob('../C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))
# %%
# CUASE 
import time 
start = time.time()
Acs = []
alpha = 0.7
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

# %%
d = 15
X, Y = se.UASE(Acs, d)
print(time.time() - start)
# %%In
# use polar coordinates to plot the embedding
def spherical_project(x):
    return x / np.linalg.norm(x)

def spherical_project_plot(x):
    d = len(x)
    theta = np.zeros(d-1)
    
    if x[0] > 0:
        theta[0] = np.arccos(x[1] / np.linalg.norm(x[:2]))
    else:
        theta[0] = 2*np.pi - np.arccos(x[1] / np.linalg.norm(x[:2]))
        
    for i in range(d-1):
        theta[i] = np.arccos(x[i+1] / np.linalg.norm(x[:(i+2)]))
    return theta
#%%
n = As[0].shape[0]
T = len(As)
#%%
XAs = np.zeros((n,d))
XAs_plot = np.zeros((n,d-1))

for i in range(n):
    if np.linalg.norm(X[i]) > 0:
        XAs[i] = spherical_project(X[i])
        XAs_plot[i] = spherical_project_plot(X[i])
#%%
YAcs = np.zeros((T,n,d))
YAcs_plot = np.zeros((T,n,d-1))
for t in range(T):
    for i in range(n):
        if np.linalg.norm(Y[t,i]) > 1e-10:
            YAcs[t,i] = spherical_project(Y[t,i])
            YAcs_plot[t,i] = spherical_project_plot(Y[t,i])
#%%
# Find all the files that start with 'labels' and end with '.pkl'
files = glob.glob('../labels/labels*.pkl')

# Load each file
labels = {}
for file in files:
    with open(file, 'rb') as f:
        labels[file] = pickle.load(f)
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

# Get all unique labels across all time points
all_labels = set()
for values in labels.values():
    all_labels.update(values)

# Fit the LabelEncoder on all unique labels
le = LabelEncoder()
le.fit(list(all_labels))

# Transform the labels at each time point
labels_encoded = {}
for key, values in labels.items():
    labels_encoded[key] = le.transform(values)

#%%


consensus_labels = np.load('consensus_labels.npy')
# %%
# %%
labels_unique = np.unique(consensus_labels)
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:len(labels_unique)])
cmap = mpl.colors.ListedColormap(colours)


#%%
# make labels_encoded a long list
labels_encoded_long = []
for key, values in labels_encoded.items():
    labels_encoded_long.extend(values)

idx = np.where(np.array(labels_encoded_long) != 7)[0]

#%%
# # # do tsne on Y
from sklearn.manifold import TSNE
YAcs_plot = YAcs_plot.reshape((T*n,d-1))
Y_embedded = TSNE(n_components=2, random_state=0).fit_transform(YAcs_plot[idx])
#%%
# separate Y_embedded
ns = [np.sum(labels_encoded[key]!= 7) for key in labels_encoded.keys()]
Y_embedded = np.split(Y_embedded, np.cumsum(ns)[:-1])

# %%
# sve Y_embedded that's a list 
import pickle

with open('Y_embedded_AUASE.pkl', 'wb') as f:
    pickle.dump(Y_embedded, f)
#%%
# load Y_embedded
with open('Y_embedded_AUASE.pkl', 'rb') as f:
    Y_embedded = pickle.load(f)

#%%
alphas = [.8,.8,.8,.8,.8,.8,.8,0.01]
#%%
# plot the embeddings in 
# a 3x4 subplot
# a 3x3 subplot
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        t = 3*i + j
        key = list(labels.keys())[t]
        labels_plot = np.array(labels[key])
        labels_plot = labels_plot[labels_plot != 'unlabeled']
        for k, label in enumerate(labels_unique):
            axs[i,j].scatter(Y_embedded[t][labels_plot == label, 0], Y_embedded[t][ labels_plot == label, 1], color=cmap(k), label=labels_unique[k], alpha=alphas[k])
            axs[i,j].set_title(key)
# add a legend

plt.legend(bbox_to_anchor=(1.5, 1), loc='lower center', fontsize='x-large')
plt.show()
# %%
