#%%
import numpy as np
from scipy import io
import xgboost as xgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
np.random.seed(0)
#%%
T = 9
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
n = As[0].shape[0]
d =15
#%%
#%%
# load active_authors.npy
active = np.load('active_authors.npy')
#%%

# Function to get positive and negative samples
def get_samples(adj_matrices, active):

    num_time_points = len(adj_matrices)
    num_nodes = adj_matrices[0].shape[0]
    
    pos_pairs = []
    neg_pairs = []
    for t in range(num_time_points - 1):
        adj_t = adj_matrices[t + 1].toarray()  # Adjacency at time t+1
        active_t = active[t]  # Active nodes at time t
        active_t_next = active[t + 1]  # Active nodes at time t+1
        
        # Positive samples: node pairs with an edge at t+1 that were active at t
        pos_pairs_t = np.array(np.where(adj_t == 1)).T
        pos_pairs_t = [pair for pair in pos_pairs_t if active_t[pair[0]] and active_t[pair[1]]]
        
        # Negative samples: node pairs that are active but don't have an edge at t+1
        all_nodes = np.arange(num_nodes)
        active_nodes_t = all_nodes[active_t & active_t_next]
        neg_pairs_t = []
        while len(neg_pairs_t) < len(pos_pairs_t):  # Balance negatives to positives
            i, j = np.random.choice(active_nodes_t, 2, replace=False)
            if adj_t[i, j] == 0:
                neg_pairs_t.append((i, j))
        
        pos_pairs.append(np.array(pos_pairs_t).reshape(-1, 2))
        neg_pairs.append(np.array(neg_pairs_t).reshape(-1, 2))

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

# load dyrep/embeddings_before_time_0_DBLP.npy to dyrep/embeddings_before_time_7_DBLP.npy
embs = [np.load(f'../DyRep/embeddings_before_time_{t+1}_DBLP.npy')[:n,] for t in range(T)]

Y = np.vstack(embs)
Y = Y.reshape(T, n, d)   
#%%


def operator_concat(u, v):
    # random shuffle u and v
    if np.random.rand() > 0.5:
        u, v = v, u
    return np.concatenate([u, v])   
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

    train_data = [generate_pair_features(pos_pairs_sample[t], neg_pairs_sample[t], Y[t], operator_concat) for t in range(T-2)]
    test_data = [generate_pair_features(pos_pairs_sample[T-2], neg_pairs_sample[T-2], Y[T-2], operator_concat)]

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

