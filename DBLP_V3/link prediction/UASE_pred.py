#%%
import numpy as np
import spectral_embedding as se
from scipy import io
import xgboost as xgb
import pickle
import glob
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
np.random.seed(0)
#%%
T = 9
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]

#%%
d = 15
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
#%%
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
# save the active.npy
np.save('active_authors.npy', active)
# %%

# Function to get positive and negative samples
import numpy as np

def get_samples(adj_matrices, active):
    num_time_points = len(adj_matrices)
    num_nodes = adj_matrices[0].shape[0]
    
    pos_pairs = []
    neg_pairs = []

    for t in range(num_time_points - 1):
        print(f"t: {t}")    
        adj_t = adj_matrices[t + 1].toarray()  # Adjacency at time t+1
        active_t = active[t]  # Active nodes at time t
        active_t_next = active[t + 1]  # Active nodes at time t+1
        
        # Positive samples: node pairs with an edge at t+1 that were active at t
        pos_pairs_t = np.array(np.where(adj_t == 1)).T
        pos_pairs_t = {tuple(pair) for pair in pos_pairs_t if active_t[pair[0]] and active_t[pair[1]]}
        
        # Negative samples: node pairs that are active but don't have an edge at t+1
        all_nodes = np.arange(num_nodes)
        active_nodes_t = all_nodes[active_t & active_t_next]
        neg_pairs_t = set()
        
        while len(neg_pairs_t) < len(pos_pairs_t):  # Balance negatives to positives
            i, j = np.random.choice(active_nodes_t, 2, replace=False)
            if adj_t[i, j] == 0:
                neg_pairs_t.add((i, j))

        # Add to the global sets
        print(f"num of pos_pairs at time {t}: {len(pos_pairs_t)}")
        print(f"num of neg_pairs at time {t}: {len(neg_pairs_t)}")

        pos_pairs.append(np.array(list(pos_pairs_t)).reshape(-1, 2))
        neg_pairs.append(np.array(list(neg_pairs_t)).reshape(-1, 2))


    return pos_pairs, neg_pairs

#%%
# Load adjacency matrices, embeddings, and active states (assumed preloaded as adj_matrices, YAcs, active)
pos_pairs, neg_pairs = get_samples(As, active)

#%%
def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0

# operator concatenation
def operator_concat(u, v):
    # random shuffle u and v
    if np.random.rand() > 0.5:
        u, v = v, u
    return np.concatenate([u, v])
#%%
# Function to generate features for a given pair of nodes
def generate_features(pair, X, operator):
    u, v = pair
    return operator(X[u], X[v])

# Function to generate features for all pairs of nodes
def generate_pair_features(pos_pairs, neg_pairs, X, operator):
    X_pairs = []
    y = []
    for pair in pos_pairs:
        X_pairs.append(generate_features(pair, X, operator))
        y.append(1)
    for pair in neg_pairs:
        X_pairs.append(generate_features(pair, X, operator))
        y.append(0)
    return np.array(X_pairs), np.array(y)
#%%
AUC = []
for i in range(10):
    # randomly sample 50% of positive and negative sample for each time point
    pos_pairs_sample = []
    neg_pairs_sample = []

    for t in range(T-1):
        pos_pairs_t = pos_pairs[t]
        neg_pairs_t = neg_pairs[t]
        pos_pairs_sample.append(pos_pairs_t[np.random.choice(len(pos_pairs_t), round(0.8*len(pos_pairs_t)), replace=False)])
        neg_pairs_sample.append(neg_pairs_t[np.random.choice(len(neg_pairs_t), round(0.8*len(neg_pairs_t)), replace=False)])

    train_data = [generate_pair_features(pos_pairs_sample[t], neg_pairs_sample[t], YAcs[t], operator_concat) for t in range(T-2)]
    test_data = [generate_pair_features(pos_pairs_sample[T-2], neg_pairs_sample[T-2], YAcs[T-2], operator_concat)]

    X_train = np.concatenate([data[0] for data in train_data])
    y_train = np.concatenate([data[1] for data in train_data])
    X_test, y_test = test_data[0]

    # Train XGBoost classifier
    clf = xgb.XGBClassifier( objective='binary:logistic')
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    AUC.append(auc)
    print(f'Iteration {i+1}: AUC = {auc}')

# %%
# compute the mean and 0.05 and 0.95 quantile of AUC
mean = np.mean(AUC)
lower = np.quantile(AUC, 0.05)
upper = np.quantile(AUC, 0.95)
mean, lower, upper

# %%
print(f"{mean:.3f}+{abs(lower-upper)/2:.3f}") 
# %%
