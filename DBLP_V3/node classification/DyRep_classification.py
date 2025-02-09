#%%
import numpy as np
import pickle
from scipy import io
import xgboost as xgb
import glob
from sklearn.metrics import accuracy_score,f1_score 
#%%
T =9
n  = 10092
d = 15

# load dyrep/embeddings_before_time_0_DBLP.npy to dyrep/embeddings_before_time_7_DBLP.npy
embs = [np.load(f'../DyRep/embeddings_before_time_{t+1}_DBLP.npy')[:n,] for t in range(T)]

Y = np.vstack(embs)
Y = Y.reshape(T, n, d)   
#%%
for t in range(0,T):
    print(t+1)
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
    active[t] = labels_encoded[key] !=7

#%%
# train the classification on 
# years 1996 to 2005 
# and test on 2006 to 2009
# i need to remove the inactive ones 
X_train = np.vstack(Y[:6])
X_train = X_train[active[:6].reshape(-1)]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[active[:6].reshape(-1)]

X_test = np.vstack(Y[6:T])
X_test = X_test[active[6:T].reshape(-1)]
y_test = np.hstack(list(labels_encoded.values())[6:T])
y_test = y_test[active[6:T].reshape(-1)]

# %%
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

#Calculating accuracy
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

#%%
# save the accuracies 
with open('accuracies_dyrep_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

# %%
# save the f1 scores
with open('f1_scores_dyrep_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
#%%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_scores_micro, 3))
print(np.round(f1_scores_macro, 3))
# %%
print(np.round(np.mean(accuracies),3))
# %%
