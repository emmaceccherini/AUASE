#%%
import numpy as np
import scipy.io as io
import pickle
import glob
import xgboost as xgb
import torch
from sklearn.metrics import accuracy_score , f1_score
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
#%%
T =9
d = 15
# load DBLP.npy
embeddings = np.load('../GloDyNe/DBLP.npy', allow_pickle = True)    
Y= [] 
for i in range(T):
    Y.append(np.vstack(list(embeddings[i].values())))
#%%
total_accuracies = np.empty((100, T))
total_f1 = np.empty((100, T))
total_f1_micro = np.empty((100, T))
total_f1_macro = np.empty((100, T))
for i in range(100):
    np.random.seed(i)
    train_idx = {}
    test_idx = {}
    for t in range(T):
        key = list(labels.keys())[t]
        n_k = Y[t].shape[0]
        perm = np.random.permutation(n_k)
        train_idx[key] = perm[:n_k//2].astype(int)
        test_idx[key] = perm[n_k//2:].astype(int)
        while True: 
            array1 = np.unique(labels_encoded[key][train_idx[key]])
            array2 = np.unique(labels_encoded[key][test_idx[key]])
            if all(element in array1 for element in array2): 
                break
            perm = np.random.permutation(n_k)
            train_idx[key] = perm[:n_k//2].astype(int)
            test_idx[key] = perm[n_k//2:].astype(int)
    for key in labels.keys():
        train_idx[key] = train_idx[key][labels_encoded[key][train_idx[key]] != 7]
        test_idx[key] = test_idx[key][labels_encoded[key][test_idx[key]] != 7]


    accuracies = []
    f1_scores = []
    f1_scores_micro = []
    f1_scores_macro = []
    for t in range(T):
        # XGBoost for classification
        key = list(labels.keys())[t]
        # at each time point 
        X_train  = Y[t][train_idx[key],:]
        X_test = Y[t][test_idx[key],:]
        y_train = labels_encoded[key][train_idx[key]]
        y_test = labels_encoded[key][test_idx[key]]
        label_encoder = LabelEncoder()

        # Fit on y_train so that classes are consistently mapped
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Number of unique classes
        k = len(label_encoder.classes_)
        model = xgb.XGBClassifier(objective='multi:softprob',
            num_class=k)

        #Training the model on the training data
        model.fit(X_train, y_train)

        #Making predictions on the test set
        predictions = model.predict(X_test )

        #Calculating accuracy
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
np.save('mean_accuracies_glodyne.npy', mean_accuracies)
np.save('mean_f1_glodyne.npy', mean_f1)
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
