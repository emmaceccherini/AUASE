# %%
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
# load DBLP.npy
embeddings = np.load('../GloDyNe/epin.npy', allow_pickle=True)    
#%%
Y= [] 
for i in range(T):
    Y.append(np.vstack(list(embeddings[i].values())))

#%%
As = [io.loadmat(f'../adj/adj_matrix_{i}.mat')['A'].astype(np.float64) for i in range(T)]
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
active = np.empty((T,n), dtype=bool)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    active[t] = labels_encoded[key] !=21
#%%
#%%
lables_new = {key: values[idx[t]] for t, (key, values) in enumerate(labels_encoded.items())}
labels_encoded = lables_new
#%%
#select from active only 
active2 = [active[t][idx[t]] for t in range(T)]
active = active2
#%%
# train the classification on 
# years 1996 to 2005 
# and test on 2006 to 2009
# i need to remove the inactive ones 

X_train = np.vstack(Y[:6])
X_train = X_train[np.concatenate(active[:6])]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[np.concatenate(active[:6])]

X_test = np.vstack(Y[6:])
X_test = X_test[np.concatenate(active[6:])]
y_test = np.hstack(list(labels_encoded.values())[6:])
y_test = y_test[np.concatenate(active[6:])]

# %%
model = xgb.XGBClassifier(objective='multi:softprob',
        num_class=21)

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test )
#%%
print(accuracy_score(y_test, predictions))  
#%%
accuracies = []
f1_scores = []
f1_scores_micro = []
f1_scores_macro = []
#Calculating accuracy
n1 = active[6].sum() 
n2 = n1 + active[7].sum()
n3 = n2 + active[8].sum()
n4 = n3 + active[9].sum()


accuracies.append(accuracy_score(y_test[:n1], predictions[:n1]))
accuracies.append(accuracy_score(y_test[n1:n2], predictions[n1:n2]))
accuracies.append(accuracy_score(y_test[n2:n3], predictions[n2:n3]))
accuracies.append(accuracy_score(y_test[n3:n4], predictions[n3:n4]))
accuracies.append(accuracy_score(y_test[n4:], predictions[n4:]))


f1_scores.append(f1_score(y_test[:n1], predictions[:n1], average='weighted'))
f1_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='weighted'))
f1_scores.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='weighted'))
f1_scores.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='weighted'))
f1_scores.append(f1_score(y_test[n4:], predictions[n4:], average='weighted'))

f1_scores_micro.append(f1_score(y_test[:n1], predictions[:n1], average='micro'))
f1_scores_micro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='micro'))
f1_scores_micro.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='micro'))
f1_scores_micro.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='micro'))
f1_scores_micro.append(f1_score(y_test[n4:], predictions[n4:], average='micro'))

f1_scores_macro.append(f1_score(y_test[:n1], predictions[:n1], average='macro'))
f1_scores_macro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='macro'))
f1_scores_macro.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='macro'))
f1_scores_macro.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='macro'))
f1_scores_macro.append(f1_score(y_test[n4:], predictions[n4:], average='macro'))
#%%
# save the accuracies 
with open('accuracies_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
#%%
# save the f1 scores
with open('f1_scores_glodyne_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
# %%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_scores_micro, 3))
print(np.round(f1_scores_macro, 3))

