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
alphas = np.linspace(0, 1, 11)
#%%
final_accuracies = []
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
    total_accuracies = []
    for iter in range(10):
        for key in labels_encoded.keys():
            n_k = len(labels_encoded[key])
            perm = np.random.permutation(n_k)
            train_idx[key] = perm[:n_k//2].astype(int)
            # select 10% of the train data as validation data
            test_idx[key] = perm[n_k//2:].astype(int)
        for key in labels_encoded.keys():
            train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 7]
            test_idx[key] = test_idx[key][labels_encoded[key][test_idx[key]] != 7]
        accuracies = []

        for t, key in enumerate(labels_encoded.keys()):
            # XGBoost for classification
            # at each time point 
            X_train  = YAcs[t][train_idx[key],:]
            X_val = YAcs[t][test_idx[key],:]
            y_train = labels_encoded[key][train_idx[key]]
            y_val = labels_encoded[key][test_idx[key]]
            model = xgb.XGBClassifier(objective='multi:softprob',
                num_class=7)
            #Training the model on the training data
            model.fit(X_train, y_train)

            #Making predictions on the test set
            predictions = model.predict(X_val) 
            accuracy = accuracy_score(y_val, predictions)
            accuracies.append(accuracy)
        print("Iteration: ", iter)
        total_accuracies.append(np.mean(accuracies))
    final_accuracies.append(total_accuracies)
    print("Alpha: ", alpha)

# %%
# %%
# save the accuracies
np.save('accuracies_sensiitivity_alpha.npy', final_accuracies)
# %%
# load the accuracies
final_accuracies = np.load('accuracies_sensiitivity_alpha.npy')
#%%
#get the 0.5 and 0.95 confidence intervals

mean_accuracies = np.mean(final_accuracies, axis=1)
accuracies_low = np.quantile(final_accuracies, 0.05, axis=1)
accuracies_high = np.quantile(final_accuracies, 0.95, axis=1)

#%%
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Create two subplots with shared x-axis for the broken y-axis effect
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5), gridspec_kw={'height_ratios': [4,1]})

# Define the break limits on the y-axis
low_y_limit = 0.59  # Lower part of the y-axis
high_y_limit = 0.85  # Upper part of the y-axis
break_y_limit_low = 0.6  # Start of the break
break_y_limit_high = 0.73  # End of the break

# Plot on the lower subplot (ax2) for the lower part of y-axis
ax2.plot(alphas, mean_accuracies, "-o", lw=3, markersize=10)
ax2.fill_between(alphas, accuracies_low, accuracies_high, alpha=0.2)

# Set y-limits for lower part
ax2.set_ylim(low_y_limit, break_y_limit_low)

# Plot on the upper subplot (ax1) for the higher part of y-axis
ax1.plot(alphas, mean_accuracies, "-o", lw=3, markersize=10)
ax1.fill_between(alphas, accuracies_low, accuracies_high, alpha=0.2)

# Set y-limits for upper part
ax1.set_ylim(break_y_limit_high, high_y_limit)

# Hide the spines between ax1 and ax2 and add break markers
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False)  # Don't show ticks on upper subplot
ax2.xaxis.tick_bottom()

# Add diagonal break markers to indicate the break in y-axis
d = 0.015  # Size of diagonal lines for break marker
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

kwargs.update(transform=ax2.transAxes)  # Switch to ax2's coordinates
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# Labels and ticks
ax2.set_xlabel('Alpha', fontsize=20)
# ax1.set_ylabel('Accuracy', fontsize=20)
# plt.set_ylabel('Accuracy', fontsize=20)
# plt.y_label('Accuracy', fontsize=20)
plt.xticks(fontsize=20)
fig.text(0.01, 0.5, 'Accuracy', va='center', ha='center', rotation='vertical', fontsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

plt.tight_layout(rect=[0, 0.0, 1, 0.97])
# Save the plot
plt.savefig('accuracy_vs_alpha_broken_axis2.png', dpi=300)

# Show the plot
# plt.show()

# %%
