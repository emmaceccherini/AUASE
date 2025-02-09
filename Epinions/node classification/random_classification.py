#%%
import numpy as np
import pandas as pd
import scipy.io as io
import spectral_embedding as se
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
np.random.seed(0)   
#%%
T = 11
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adj/adj_matrix_{i}.mat')['A'].astype(np.float64) for i in range(T)]


# %%
# load label_encoded.py
labels_encoded = np.load('../labels_encoded.npy', allow_pickle=True).item()
#reorder the labels_encoded
#%%
#get all the labels
labels = np.unique(np.concatenate([labels_encoded[key] for key in labels_encoded.keys()]))
#%%
new_max_label = labels.max() + 1
labels[labels == 25] = new_max_label
labels = np.sort(labels)
label_mapping = {label: idx for idx, label in enumerate(labels)}
#%%
label_mapping[25]= 21
#%%
# Step 3: Apply the mapping to each year in the dictionary
new_labels_dict = {}
for year, labels in labels_encoded.items():
    # Apply the remapping to the labels
    new_labels_dict[year] = np.array([label_mapping[label] for label in labels])
#%%
labels_encoded = new_labels_dict

#%%
most_popular_class = np.zeros(T)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    most_popular_class[t] = np.argmax(np.bincount(labels_encoded[key][labels_encoded[key] != 21]))
#%%
# get the size of the most popular class that is not
# 7
size_most_popular_class = np.zeros(T)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    size_most_popular_class[t] = np.sum(labels_encoded[key] == most_popular_class[t])
#%%
# get the sum of the sizes of all classes that are not 7
size_all_classes = np.zeros(T)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    size_all_classes[t] = np.sum(labels_encoded[key] != 21)
#%%
acc  = size_most_popular_class/size_all_classes
#%%
# save acc
with open('acc_random.pkl', 'wb') as f:
    pickle.dump(acc, f)
#%%