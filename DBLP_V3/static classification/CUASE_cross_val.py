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
import glob
#%%
# load adj_matrix_0.mat to adj_matrix_11.mat
#%% CHECK SHOULD BE 10 
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(9)]

#%%
# Get a list of all files that match the pattern
files = glob.glob('../C_matrices/word_count_matrix_*.npz')

Cs = []
for file in files:
    Cs.append(sparse.load_npz(file))
#%%
n = As[0].shape[0]
T = len(As)
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
labels_encoded  ={}
le = LabelEncoder()
for key, values in labels.items():
    labels_encoded[key] = le.fit_transform(labels[key])


#%%
long_labels = np.hstack(list(labels_encoded.values()))
#%%
alphas = np.linspace(0.1, 0.9, 9)
total_accuracies = np.zeros((len(alphas), len(labels_encoded.keys())))
vald_idx_AUASE = {}
for a,alpha in enumerate(alphas):
    Acs = []
    for i in range(len(As)):
        A = As[i]
        C = Cs[i]
        p = Cs[0].shape[1]
        # standardize the columns of C
        C = normalize(C, axis=0)
        top = sparse.hstack([(1-alpha)*A, alpha * C])
        bottom = sparse.hstack([alpha * C.T, sparse.csr_matrix((p, p))])
        Ac = sparse.vstack([top, bottom])
        Acs.append(Ac)
    d = 15
    X, Y = se.UASE(Acs, d)

    YAcs = np.zeros((T,n,d))
    for t in range(T):
        for i in range(n):
            if np.linalg.norm(Y[t,i]) > 1e-10:
                YAcs[t,i] = spherical_project(Y[t,i])
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
        # remove the validation data from the training data
        train_idx[key] = np.setdiff1d(train_idx[key], val_idx[key])
    # save the validation data
    vald_idx_AUASE[alpha] = val_idx
    for key in labels_encoded.keys():
        train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 7]
        val_idx[key] = val_idx[key][labels_encoded[key][val_idx[key]] != 7]
    accuracies = []

    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        # at each time point 
        X_train  = YAcs[t][train_idx[key],:]
        X_val = YAcs[t][val_idx[key],:]
        y_train = labels_encoded[key][train_idx[key]]
        y_val = labels_encoded[key][val_idx[key]]
        model = xgb.XGBClassifier(objective='multi:softprob',
            num_class=7)
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
# save the validation data
np.save('AUASE_val_idx.npy', vald_idx_AUASE[alpha])
# %%
