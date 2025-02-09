#%%
import numpy as np
import spectral_embedding as se
from scipy import io
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, f1_score
import glob
#%%
T = 15
np.random.seed(0)

#%%
files = glob.glob('../labels/labels*.pkl')

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
from sklearn.preprocessing import LabelEncoder
labels_encoded  ={}
le = LabelEncoder()
for key, values in labels.items():
    labels_encoded[key] = le.fit_transform(labels[key])
#%%
# get teh most popular class that is not 7
most_popular_class = np.zeros(T)
for t in range(T):
    key = list(labels.keys())[t]
    most_popular_class[t] = np.argmax(np.bincount(labels_encoded[key][labels_encoded[key] != 2]))
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
    size_all_classes[t] = np.sum(labels_encoded[key] != 2)
#%%
acc  = size_most_popular_class/size_all_classes
#%%
# save acc
with open('acc_random.pkl', 'wb') as f:
    pickle.dump(acc, f)
#%%