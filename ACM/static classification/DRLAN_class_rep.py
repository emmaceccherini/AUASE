# %%
import numpy as np
import spectral_embedding as se
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score,f1_score
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from scipy import io
import sklearn.model_selection
import glob
# %%
T = 15
# %%
# Find all the files that start with 'labels' and end with '.pkl'
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
n  = 34339
d = 29
alpha = 0.3
embs = [io.loadmat(f'../DRLAN/embedding_{t+1}beta{alpha}.mat')['U'] for t in range(T)] 
Y = np.vstack(embs)
Y = Y.reshape(T, n, d)  
#%%
val_idx = np.load('DRLAN_val_idx.npy', allow_pickle=True).item()
non_val_idx = {}
for key in labels_encoded.keys():
    non_val_idx[key] = np.array(list(set(range(n)) - set(val_idx[key])))
#%%
total_accuracies = np.empty((100, len(labels.keys())))
total_f1 = np.empty((100, len(labels.keys())))
total_f1_micro = np.empty((100, len(labels.keys())))
total_f1_macro = np.empty((100, len(labels.keys())))
for i in range(100):
    np.random.seed(i)
    train_idx = {}
    test_idx = {}
    for key in labels_encoded.keys():
        n_k = len(labels_encoded[key])
        perm = np.random.permutation(n_k)
        # remove the validation data from the training data
        perm = perm[non_val_idx[key]]
        train_idx[key] = perm[:n_k//2].astype(int)
        test_idx[key] = perm[n_k//2:].astype(int)
    for key in labels_encoded.keys():
        train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 2]
        test_idx[key] = test_idx[key][labels_encoded[key][test_idx[key]] != 2]
    accuracies = []
    f1_scores = []
    f1_scores_micro = []
    f1_scores_macro = []
    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        # at each time point 
        X_train  = Y[t][train_idx[key],:]
        X_test = Y[t][test_idx[key],:]
        y_train = labels_encoded[key][train_idx[key]]
        y_test = labels_encoded[key][test_idx[key]]
        model = xgb.XGBClassifier(objective='binary:logistic')

        #Training the model on the training data
        model.fit(X_train, y_train)

        #Making predictions on the test set
        predictions = model.predict(X_test )
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        f1_scores.append(f1_score(y_test, predictions, average='weighted'))
        f1_scores_micro.append(f1_score(y_test, predictions, average='micro'))
        f1_scores_macro.append(f1_score(y_test, predictions, average='macro'))
    total_accuracies[i] = accuracies
    total_f1[i] = f1_scores
    total_f1_micro[i] = f1_scores_micro 
    total_f1_macro[i] = f1_scores_macro
    print("Iteration: ", i)
#%%
#%%
# save the mean accuracies
mean_accuracies = np.mean(total_accuracies, axis=0)
mean_f1 = np.mean(total_f1, axis=0)
mean_f1_micro = np.mean(total_f1_micro, axis=0)
mean_f1_macro = np.mean(total_f1_macro, axis=0)

accuracies_low = np.quantile(total_accuracies, 0.05, axis=0)
accuracies_high = np.quantile(total_accuracies, 0.95, axis=0)
f1_low = np.quantile(total_f1, 0.05, axis=0)
f1_high = np.quantile(total_f1, 0.95, axis=0)
f1_micro_low = np.quantile(total_f1_micro, 0.05, axis=0)
f1_micro_high = np.quantile(total_f1_micro, 0.95, axis=0)
f1_macro_low = np.quantile(total_f1_macro, 0.05, axis=0)
f1_macro_high = np.quantile(total_f1_macro, 0.95, axis=0)
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
#%%
# save the mean accuracies
np.save('mean_accuracies_DRLAN.npy', mean_accuracies)
np.save('mean_f1_DRLAN.npy', mean_f1)
# %%
