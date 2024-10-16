#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral_embedding as se
from scipy import io
import scipy.sparse as sparse
import pickle
import matplotlib as mpl
import glob
#%%
n = 10092 
# Find all the files that start with 'labels' and end with '.pkl'
files = glob.glob('../labels/labels*.pkl')

# Load each file
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
labels_encoded  ={}
le = LabelEncoder()
for key, values in labels.items():
    labels_encoded[key] = le.fit_transform(labels[key])
#%%
# get "consensus labels"
from collections import defaultdict
import statistics

consensus_labels = []
for i in range(n):
    author_labels = []
    for key, value in labels.items():
        if value[i] != "unlabeled":
            author_labels.append(value[i])
    consensus_labels.append(statistics.mode(author_labels))
# %%
labels_unique = np.unique(consensus_labels)
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:len(labels_unique)])
cmap = mpl.colors.ListedColormap(colours)
#%%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels_encoded = le.fit_transform(consensus_labels)

#%%
# save consensus labels
np.save('consensus_labels.npy', labels_encoded)