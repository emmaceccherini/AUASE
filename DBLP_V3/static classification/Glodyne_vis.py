#%%
import numpy as np
import scipy.io as io
import pickle
import glob
import xgboost as xgb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import torch
from sklearn.metrics import accuracy_score , f1_score
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
# get "consensus labels"
# load consensus_labels.npy

consensus_labels = np.load('consensus_labels.npy')
# %%
# %%
labels_unique = np.unique(consensus_labels)
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:len(labels_unique)])
cmap = mpl.colors.ListedColormap(colours)
#%%
#%%
T =9
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
idx = [np.where(np.sum(As[t], axis=1) != 0)[0] for t in range(T)]
n = As[0].shape[0]
#%%
# modify the idx so that idx 2 is the sum of idx 2 plus idx 1 
idx[1] = np.unique(np.concatenate([idx[0], idx[1]]))
idx[2] = np.unique(np.concatenate([idx[1], idx[2]]))
idx[3] = np.unique(np.concatenate([idx[2], idx[3]]))
idx[4] = np.unique(np.concatenate([idx[3], idx[4]]))
idx[5] = np.unique(np.concatenate([idx[4], idx[5]]))
idx[6] = np.unique(np.concatenate([idx[5], idx[6]]))
idx[7] = np.unique(np.concatenate([idx[6], idx[7]]))
idx[8] = np.unique(np.concatenate([idx[7], idx[8]]))
# %%
lables_new = {key: np.array(values)[idx[t]] for t, (key, values) in enumerate(labels.items())}
labels = lables_new
#%%
# do the same for enconded labels
labels_encoded_new = {key: np.array(values)[idx[t]] for t, (key, values) in enumerate(labels_encoded.items())}
labels_encoded = labels_encoded_new
#%%
alphas = [.8,.8,.8,.8,.8,.8,.8,0.1]

#%%
d = 15
# load DBLP.npy
embeddings = np.load('../GloDyNe/DBLP.npy', allow_pickle = True)    
Y= [] 
for i in range(T):
    Y.append(np.vstack(list(embeddings[i].values())))

#%%
Y_embedded = []
for t in range(T):
    key = list(labels.keys())[t]
    # idx = np.where(np.array(labels[key]) != "unlabeled")[0]
    Y_embedded.append(TSNE(n_components=2, random_state=1).fit_transform(Y[t]))
    print(f"Finished TSNE for t={t}")
# %%
#%%
import pickle

with open('Y_embedded_glodyne.pkl', 'wb') as f:
    pickle.dump(Y_embedded, f)
# %%
alphas = [.8,.8,.8,.8,.8,.8,.8,0.1]

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
        # labels_plot = labels_plot[labels_plot != 'unlabeled']
        for k, label in enumerate(labels_unique):
            axs[i,j].scatter(Y_embedded[t][labels_plot == label, 0], Y_embedded[t][ labels_plot == label, 1], color=cmap(k), label=labels_unique[k], alpha=alphas[k])
            axs[i,j].set_title(key)
# add a legend

plt.legend(bbox_to_anchor=(1.5, 1), loc='lower center', fontsize='x-large')
plt.show()

 # %%

# %%
# do tsne together
labels_encoded_long = []
for key, values in labels_encoded.items():
    labels_encoded_long.extend(values)

idx = np.where(np.array(labels_encoded_long) != 7)[0]
#%%
# # # do tsne on Y
from sklearn.manifold import TSNE
Y_embedded = TSNE(n_components=2,
                  random_state=0).fit_transform(np.vstack(Y)[idx])
#%%
# separate Y_embedded
ns = [np.sum(labels_encoded[key]!= 7) for key in labels_encoded.keys()]
Y_embedded = np.split(Y_embedded, np.cumsum(ns)[:-1])

# %%


with open('Y_embedded_glodyne_long.pkl', 'wb') as f:
    pickle.dump(Y_embedded, f)
# %%
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
