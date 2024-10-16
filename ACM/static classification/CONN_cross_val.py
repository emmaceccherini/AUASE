#%%
import numpy as np
import spectral_embedding as se
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score
import sklearn
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from scipy import io
import torch
import glob
#%%
#%%
# Get a list of all files that match the pattern
files = glob.glob('../C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))
#%%
n = Cs[0].shape[0]
T = len(Cs)

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
labels_encoded  ={}
le = LabelEncoder()
for key, values in labels.items():
    labels_encoded[key] = le.fit_transform(labels[key])

#%%
# do cross validation to choose alpha 
# get 10% splits of the data
long_labels = np.hstack(list(labels_encoded.values()))
#%%
d = 29
vald_idx_CONN = {}
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
total_accuracies = np.zeros((len(alphas), len(labels_encoded.keys())))
for a,alpha in enumerate(alphas):
    embs = [torch.load(f'../CONN/embeddings_lp_ACM_{t}_tw{alpha}.pth') for t in range(T)] 
    Y = np.vstack(embs)
    Y = Y.reshape(T, n, d)   

    train_idx = {}
    test_idx = {}
    val_idx = {}
    for key in labels_encoded.keys():
        n_k = len(labels_encoded[key])
        perm = np.random.permutation(n_k)
        train_idx[key] = perm[:n_k//2].astype(int)
        # select 10% of the train data as validation data
        perm_val = np.random.permutation(train_idx[key])
        val_idx[key] = perm_val[:len(perm_val)//10].astype(int)
        test_idx[key] = perm[n_k//2:].astype(int)
        train_idx[key] = np.setdiff1d(train_idx[key], val_idx[key])

    # save the validation data
    vald_idx_CONN[alpha] = val_idx
    for key in labels_encoded.keys():
        train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 2]
        val_idx[key] = val_idx[key][labels_encoded[key][val_idx[key]] != 2]
        test_idx[key] = test_idx[key][labels_encoded[key][test_idx[key]] != 2]
    model = xgb.XGBClassifier(objective= "binary:logistic")
    accuracies = []

    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        # at each time point 
        X_train  = Y[t][train_idx[key],:]
        X_test = Y[t][test_idx[key],:]
        X_val = Y[t][val_idx[key],:]
        y_train = labels_encoded[key][train_idx[key]]
        y_test = labels_encoded[key][test_idx[key]]
        y_val = labels_encoded[key][val_idx[key]]
        model = xgb.XGBClassifier(objective='binary:logistic')

        #Training the model on the training data
        model.fit(X_train, y_train)

        #Making predictions on the test set
        predictions = model.predict(X_val) 
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)
    total_accuracies[a] = accuracies
    print("Alpha: ", alpha)
# %%
alpha = alphas[np.argmax(np.mean(total_accuracies, axis =1))]
print(alpha)
# %%
np.save('CONN_val_idx.npy', vald_idx_CONN[alpha])