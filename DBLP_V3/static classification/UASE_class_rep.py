#%%
import numpy as np
import spectral_embedding as se
from scipy import io
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, f1_score
import glob
np.random.seed(0)
#%%
T = 9
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'../adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]

#%%
d = 15
X, Y = se.UASE(As, d)
#%%
# save X and Y
np.save('X_UASE.npy', X)
np.save('Y_UASE.npy', Y)
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
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import time  # Import the time module

# Assuming labels, labels_encoded, and YAcs are already defined

total_accuracies = np.empty((100, len(labels.keys())))
total_f1 = np.empty((100, len(labels.keys())))
total_f1_micro = np.empty((100, len(labels.keys())))
total_f1_macro = np.empty((100, len(labels.keys())))
iteration_time = []  # To store the time taken for each iteration
for i in range(100):
    start_time = time.time()  # Start timing the iteration
    
    # reset the seed 
    np.random.seed(i)
    train_idx = {}
    test_idx = {}
    for key in labels.keys():
        n_k = len(labels[key])
        perm = np.random.permutation(n_k)
        train_idx[key] = perm[:n_k//2].astype(int)
        test_idx[key] = perm[n_k//2:].astype(int)
    
    for key in labels.keys():
        train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 7]
        test_idx[key] = test_idx[key][labels_encoded[key][test_idx[key]] != 7]
    
    accuracies = []
    f1_scores = []
    f1_micro =[]
    f1_macro = []
    for t, key in enumerate(labels.keys()):
        # XGBoost for classification
        X_train = YAcs[t][train_idx[key], :]
        X_test = YAcs[t][test_idx[key], :]
        y_train = labels_encoded[key][train_idx[key]]
        y_test = labels_encoded[key][test_idx[key]]
        
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=7)

        # Training the model on the training data
        model.fit(X_train, y_train)

        # Making predictions on the test set
        predictions = model.predict(X_test)

        # Calculating accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        f1_scores.append(f1_score(y_test, predictions, average='weighted'))
        f1_micro.append(f1_score(y_test, predictions, average='micro'))
        f1_macro.append(f1_score(y_test, predictions, average='macro'))
    
    total_accuracies[i] = accuracies
    total_f1[i] = f1_scores
    total_f1_micro[i] = f1_micro
    total_f1_macro[i] = f1_macro
    
    end_time = time.time()  # End timing the iteration
    iteration_time.append(end_time - start_time) # Calculate the time taken
    
    print(f"Iteration {i} completed in {iteration_time[i]:.4f} seconds")


#%%
# do the mean over the 10 iterations
mean_accuracies = np.mean(total_accuracies, axis=0)
mean_f1 = np.mean(total_f1, axis=0)
mean_f1_micro = np.mean(total_f1_micro, axis=0)
mean_f1_macro = np.mean(total_f1_macro, axis=0)
#%%
#get the 0.5 and 0.95 confidence intervals
accuracies_low = np.quantile(total_accuracies, 0.05, axis=0)
accuracies_high = np.quantile(total_accuracies, 0.95, axis=0)
f1_low = np.quantile(total_f1, 0.05, axis=0)
f1_high = np.quantile(total_f1, 0.95, axis=0)
f1_micro_low = np.quantile(total_f1_micro, 0.05, axis=0)
f1_micro_high = np.quantile(total_f1_micro, 0.95, axis=0)
f1_macro_low = np.quantile(total_f1_macro, 0.05, axis=0)
f1_macro_high = np.quantile(total_f1_macro, 0.95, axis=0)
#%%

# save the mean accuracies
np.save('mean_accuracies_UASE.npy', mean_accuracies)
np.save('mean_f1_UASE.npy', mean_f1)
#%%
#%%
print(f"Mean accuracy: {np.mean(mean_accuracies):.3f}")
print(f"Mean F1 score: {np.mean(mean_f1):.3f}")
print(f"Mean F1 score (micro): {np.mean(mean_f1_micro):.3f}")
print(f"Mean F1 score (macro): {np.mean(mean_f1_macro):.3f}")
# print the length of the confidence interval
print(f"Acc Confidence interval length 0.5: {(np.mean(accuracies_high) - np.mean(accuracies_low))/2:.3f}")
print(f"F1 Confidence interval length 0.5: {(np.mean(f1_high) - np.mean(f1_low))/2:.3f}")
print(f"F1 Confidence interval length 0.5 (micro): {(np.mean(f1_micro_high) - np.mean(f1_micro_low))/2:.3f}")
print(f"F1 Confidence interval length 0.5 (macro): {(np.mean(f1_macro_high) - np.mean(f1_macro_low))/2:.3f}")

#%%
print("long version")
print(f"Mean accuracy: {np.round(mean_accuracies,3)}")
print(f"Mean F1 score: {np.round(mean_f1,3)}")
print(f"Mean F1 score (micro): {np.round(mean_f1_micro,3)}")
print(f"Mean F1 score (macro): {np.round(mean_f1_macro,3)}")
print(f"Acc Confidence interval length 0.5: {np.round((accuracies_high - accuracies_low)/2,3)}")
print(f"F1 Confidence interval length 0.5: {np.round((f1_high - f1_low)/2,3)}")
print(f"F1 Confidence interval length 0.5 (micro): {np.round((f1_micro_high - f1_micro_low)/2,3)}")
print(f"F1 Confidence interval length 0.5 (macro): {np.round((f1_macro_high - f1_macro_low)/2,3)}")
# %%
