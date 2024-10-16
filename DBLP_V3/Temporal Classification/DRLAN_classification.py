#%%
import numpy as np
import pickle
from scipy import io
import xgboost as xgb
import glob
from sklearn.metrics import accuracy_score  ,f1_score
#%%
T =9
n  = 10092 
d = 15
#%%
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
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

accuracies_val = []
val_idx = {}
for a, alpha in enumerate(alphas):
    embs = [io.loadmat(f'../DRLAN/embedding_{t+1}beta{alpha}.mat')['U'] for t in range(T)] 
    Y = np.vstack(embs)
    Y = Y.reshape(T, n, d)   

    X_train = np.vstack(Y[:6])
    X_train = X_train[active[:6].reshape(-1)]
    y_train = np.hstack(list(labels_encoded.values())[:6])
    y_train = y_train[active[:6].reshape(-1)]
    # select as validation data 10% of the training data
    n_val = int(0.1 * X_train.shape[0])
    perm = np.random.permutation(X_train.shape[0])
    val_idx[alpha] = perm[:n_val]
    X_val = X_train[perm[:n_val]]
    y_val = y_train[perm[:n_val]]
    
    # remove the validation data from the training data
    X_train = X_train[perm[n_val:]]
    y_train = y_train[perm[n_val:]]


    model = xgb.XGBClassifier(objective='multi:softprob',
            num_class=7)

    #Training the model on the training data
    model.fit(X_train, y_train)

    #Making predictions on the test set
    predictions = model.predict(X_val)

    #Calculating accuracy
    accuracy = accuracy_score(y_val, predictions)
    # print(f"Accuracy: {accuracy}")
    #save the accuracy for each time point
    accuracies_val.append(accuracy )  
#%%
# save the accuracies for each alpha
with open('accuracies_val_DRLAN.pkl', 'wb') as f:
    pickle.dump(accuracies_val, f)
#%%
alpha = alphas[np.argmax(accuracies_val)]
print(alpha)
#%%
accuracies_test = []
embs = [io.loadmat(f'../DRLAN/embedding_{t+1}beta{alpha}.mat')['U'] for t in range(T)] 
Y = np.vstack(embs)
Y = Y.reshape(T, n, d)  
#%%
# XGBoost for classification
# at each time point 
X_train = np.vstack(Y[:6])
X_train = X_train[active[:6].reshape(-1)]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[active[:6].reshape(-1)]
#remove the validation data from the training data
X_train = np.delete(X_train, val_idx[alpha], axis=0)
y_train = np.delete(y_train, val_idx[alpha], axis=0)

X_test = np.vstack(Y[6:])
X_test = X_test[active[6:].reshape(-1)]
y_test = np.hstack(list(labels_encoded.values())[6:])
y_test = y_test[active[6:].reshape(-1)]

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

# %%
with open('accuracies_DRLAN_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
#%%
np.round(accuracies, 3)
# %%
with open('f1_scores_DRLAN_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
#%%
np.round(f1_scores, 3)    
#%%
np.round(f1_scores_micro, 3)
#%%
np.round(f1_scores_macro, 3)
