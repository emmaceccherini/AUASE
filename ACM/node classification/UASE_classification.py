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
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adjs/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
np.random.seed(0)
#%%
d = 29
X, Y = se.UASE(As, d)

# %%
# use polar coordinates to plot the embedding
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
#%%
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
# save the active authors 
active = np.empty((T,n), dtype=bool)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    active[t] = labels_encoded[key] !=2

#%% 
X_train = np.vstack(YAcs[:10])
X_train = X_train[active[:10].reshape(-1)]
y_train = np.hstack(list(labels_encoded.values())[:10])
y_train = y_train[active[:10].reshape(-1)]

X_test = np.vstack(YAcs[10:])
X_test = X_test[active[10:].reshape(-1)]
y_test = np.hstack(list(labels_encoded.values())[10:])
y_test = y_test[active[10:].reshape(-1)]

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
with open('accuracies_UASE_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
#%%
# save the f1 scores
with open('f1_scores_UASE_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
# %%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_micro_scores, 3))
print(np.round(f1_macro_scores, 3))

# %%
