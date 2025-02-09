#%%
import numpy as np
import spectral_embedding as se
from scipy import io
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score,f1_score
import glob
np.random.seed(1)
#%%
T = 9
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
n = As[0].shape[0]
#%%
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
# get teh most popular class that is not 7
most_popular_class = np.zeros(T)
for t in range(T):
    key = list(labels.keys())[t]
    most_popular_class[t] = np.argmax(np.bincount(labels_encoded[key][labels_encoded[key] != 7]))
#%%
# get the size of the most popular class that is not
# 7
size_most_popular_class = np.zeros(T)
for t in range(T):
    key = list(labels.keys())[t]
    size_most_popular_class[t] = np.sum(labels_encoded[key] == most_popular_class[t])
#%%
# get the sum of the sizes of all classes that are not 7
size_all_classes = np.zeros(T)
for t in range(T):
    key = list(labels.keys())[t]
    size_all_classes[t] = np.sum(labels_encoded[key] != 7)
#%%
acc  = size_most_popular_class/size_all_classes
#%%
# save acc
with open('acc_random.pkl', 'wb') as f:
    pickle.dump(acc, f)
#%%
