#%%
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, f1_score
import glob
import scipy.io as io
import torch
#%%
T =9
d = 15
n = 10092

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
T =9
d = 15
# load DBLP.npy
embeddings = np.load('../GloDyNE/DBLP.npy', allow_pickle=True)    
Y= [] 
for i in range(T):
    Y.append(np.vstack(list(embeddings[i].values())))
#%%
#%%
As = [io.loadmat(f'adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
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
lables_new = {key: values[idx[t]] for t, (key, values) in enumerate(labels_encoded.items())}
labels_encoded = lables_new
#%%
active = []
for t in range(T):
    key = list(labels.keys())[t]
    active.append( labels_encoded[key] !=7)
#%%
# XGBoost for classification
# at each time point 
X_train = np.vstack(Y[:6])
X_train = X_train[np.concatenate(active[:6])]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[np.concatenate(active[:6])]

X_test = np.vstack(Y[6:])
X_test = X_test[np.concatenate(active[6:])]
y_test = np.hstack(list(labels_encoded.values())[6:])
y_test = y_test[np.concatenate(active[6:])]

#%%
model = xgb.XGBClassifier(objective='multi:softprob',
        num_class=7)

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test )

#%%
accuracies = []
f1_scores = []
f1_scores_micro = []
f1_scores_macro = []
n1 = active[6].sum()
n2 = active[7].sum() +n1


accuracies.append(accuracy_score(y_test[:n1], predictions[:n1]))
accuracies.append(accuracy_score(y_test[n1:n2], predictions[n1:n2]))
accuracies.append(accuracy_score(y_test[n2:], predictions[n2:]))
# accuracies.append(accuracy_score(y_test[n3:n4], predictions[n3:n4]))
# accuracies.append(accuracy_score(y_test[n4:], predictions[n4:]))

f1_scores.append(f1_score(y_test[:n1], predictions[:n1], average='weighted'))
f1_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='weighted'))
f1_scores.append(f1_score(y_test[n2:], predictions[n2:], average='weighted'))


f1_scores_micro.append(f1_score(y_test[:n1], predictions[:n1], average='micro'))
f1_scores_micro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='micro'))
f1_scores_micro.append(f1_score(y_test[n2:], predictions[n2:], average='micro'))

f1_scores_macro.append(f1_score(y_test[:n1], predictions[:n1], average='macro'))
f1_scores_macro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='macro'))
f1_scores_macro.append(f1_score(y_test[n2:], predictions[n2:], average='macro'))

# %%
with open('accuracies_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
# %%
with open('f1_scores_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
#%%
np.round(accuracies, 3)
# %%
np.round(f1_scores, 3)
# %%
np.round(f1_scores_micro, 3)
# %%
np.round(f1_scores_macro, 3)
# %%
