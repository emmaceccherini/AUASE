#%%
import numpy as np
import pickle
from scipy import io
import xgboost as xgb
import glob
from sklearn.metrics import accuracy_score, f1_score

#%%
T =15
embeddings = np.load("../GloDyNe/ACM.npy",
                     allow_pickle=True)
Y= [] 
for i in range(T):
    Y.append(np.vstack(list(embeddings[i].values())))
#%%
As = [io.loadmat(f'../adjs/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
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
idx[9] = np.unique(np.concatenate([idx[8], idx[9]]))
idx[10] = np.unique(np.concatenate([idx[9], idx[10]]))
idx[11] = np.unique(np.concatenate([idx[10], idx[11]]))
idx[12] = np.unique(np.concatenate([idx[11], idx[12]]))
idx[13] = np.unique(np.concatenate([idx[12], idx[13]]))
idx[14] = np.unique(np.concatenate([idx[13], idx[14]]))
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
# save the active authors 
active = np.empty((T,n), dtype=bool)
for t in range(T):
    key = list(labels.keys())[t]
    active[t] = labels_encoded[key] !=2
#%%
# %%
lables_new = {key: values[idx[t]] for t, (key, values) in enumerate(labels_encoded.items())}
labels_encoded = lables_new
#%%
active2 = [active[t][idx[t]] for t in range(T)]
active = active2
#%%
X_train = np.vstack(Y[:10])
X_train = X_train[np.concatenate(active[:10])]
y_train = np.hstack(list(labels_encoded.values())[:10])
y_train = y_train[np.concatenate(active[:10])]

X_test = np.vstack(Y[10:])
X_test = X_test[np.concatenate(active[10:])]
y_test = np.hstack(list(labels_encoded.values())[10:])
y_test = y_test[np.concatenate(active[10:])]
# %%
model = xgb.XGBClassifier(objective='binary:logistic')

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test )

#%%
accuracies = []
f1_scores = []
f1_micro_scores = []
f1_macro_scores = []

#Calculating accuracy
n1 = active[10].sum() 
n2 = n1 + active[11].sum()
n3 = n2 + active[12].sum()
n4 = n3 + active[13].sum()
# n5 = n4 + active[14].sum()

accuracies.append(accuracy_score(y_test[:n1], predictions[:n1]))
accuracies.append(accuracy_score(y_test[n1:n2], predictions[n1:n2]))
accuracies.append(accuracy_score(y_test[n2:n3], predictions[n2:n3]))
accuracies.append(accuracy_score(y_test[n3:n4], predictions[n3:n4]))
accuracies.append(accuracy_score(y_test[n4:], predictions[n4:]))
# accuracies.append(accuracy_score(y_test[n5:], predictions[n5:]))

f1_scores.append(f1_score(y_test[:n1], predictions[:n1], average='weighted'))
f1_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='weighted'))
f1_scores.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='weighted'))
f1_scores.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='weighted'))
f1_scores.append(f1_score(y_test[n4:], predictions[n4:], average='weighted'))
# f1_scores.append(f1_score(y_test[n5:], predictions[n5:], average='weighted'))

f1_micro_scores.append(f1_score(y_test[:n1], predictions[:n1], average='micro'))
f1_micro_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='micro'))
f1_micro_scores.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='micro'))
f1_micro_scores.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='micro'))
f1_micro_scores.append(f1_score(y_test[n4:], predictions[n4:], average='micro'))


f1_macro_scores.append(f1_score(y_test[:n1], predictions[:n1], average='macro'))
f1_macro_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='macro'))
f1_macro_scores.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='macro'))
f1_macro_scores.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='macro'))
f1_macro_scores.append(f1_score(y_test[n4:], predictions[n4:], average='macro'))
#%%
# save the accuracies 
with open('accuracies_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

# %%
# save the f1 scores
with open('f1_scores_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)

# save the f1 micro scores
with open('f1_micro_scores_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(f1_micro_scores, f)

# save the f1 macro scores
with open('f1_macro_scores_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(f1_macro_scores, f) 
#%%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_micro_scores, 3))
print(np.round(f1_macro_scores, 3))
# %%
