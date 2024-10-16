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
d = 22
import time
start = time.time()
X, Y = se.UASE(As, d)
print(time.time() - start)
#%%
np.save('X_UASE.npy', X)
np.save('Y_UASE.npy', Y)
# %%
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
# %%
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
X_train = np.vstack(YAcs[:6])
X_train = X_train[active[:6].reshape(-1)]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[active[:6].reshape(-1)]

X_test = np.vstack(YAcs[6:])
X_test = X_test[active[6:].reshape(-1)]
y_test = np.hstack(list(labels_encoded.values())[6:])
y_test = y_test[active[6:].reshape(-1)]
#%%
model = xgb.XGBClassifier(objective='multi:softprob',
        num_class=21)

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
with open('accuracies_UASE_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
#%%
# save the f1 scores
with open('f1_scores_UASE_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
# %%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_scores_micro, 3))
print(np.round(f1_scores_macro, 3))

# %%
