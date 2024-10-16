#%%
import numpy as np
import spectral_embedding as se
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from scipy import io
from sklearn.preprocessing import LabelEncoder
import time
np.random.seed(0)
#%%
T = 11
As = [io.loadmat(f'../adj/adj_matrix_{i}.mat')['A'].astype(np.float64) for i in range(T)]
Cs = [io.loadmat(f'../Cs/C_{i}.mat')['C'].astype(np.float64) for i in range(T)]
#%%
n = As[0].shape[0]
#%%
#get all the labels
# load label_encoded.py
labels_encoded = np.load('../labels_encoded.npy', allow_pickle=True).item()
#reorder the labels_encoded
#%%
#get all the labels
labels = np.unique(np.concatenate([labels_encoded[key] for key in labels_encoded.keys()]))
#%%
new_max_label = labels.max() + 1
labels[labels == 25] = new_max_label
labels = np.sort(labels)
label_mapping = {label: idx for idx, label in enumerate(labels)}
#%%
label_mapping[25]= 21
#%%
# Step 3: Apply the mapping to each year in the dictionary
new_labels_dict = {}
for year, labels in labels_encoded.items():
    # Apply the remapping to the labels
    new_labels_dict[year] = np.array([label_mapping[label] for label in labels])
#%%
labels_encoded = new_labels_dict
# %%
import time
start = time.time()
alpha = 0.7
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
d = 22
X, Y = se.UASE(Acs, d)
print(time.time() - start)
#%%
np.save('X_AUASE.npy', X)
np.save('Y_AUASE.npy', Y)
#%%
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
YAcs = np.zeros((T,n,d))
for t in range(T):
    for i in range(n):
        if np.linalg.norm(Y[t,i]) > 1e-10:
            YAcs[t,i] = spherical_project(Y[t,i])
#%%
# load 'AUASE_val_idx.npy'
vald_idx_AUASE = np.load('AUASE_val_idx.npy', allow_pickle=True).item()

#%%

total_accuracies = np.empty((100, len(labels_encoded.keys())))
total_f1 = np.empty((100, len(labels_encoded.keys())))
total_f1_micro = np.empty((100, len(labels_encoded.keys())))
total_f1_macro = np.empty((100, len(labels_encoded.keys())))
iteration_time = []  # To store the time taken for each iteration
for i in range(100):
    np.random.seed(i)
    train_idx = {}
    test_idx = {}

    for key in labels_encoded.keys():
        # Get unique classes in the current label set
        unique_classes = np.unique(labels_encoded[key])
        
        # Create a list to hold training and testing indices
        train_idx[key] = []
        test_idx[key] = []
        
        for class_label in unique_classes:
            if class_label == 21:  # Skip class 25 if it's not relevant
                continue
            
            # Get indices for the current class
            class_indices = np.where(labels_encoded[key] == class_label)[0]
            
            # Shuffle the indices of this class
            np.random.shuffle(class_indices)
            
            # Split the class indices into train and test
            n_class = len(class_indices)
            train_size = max(1, n_class // 2)  # Ensure at least 1 sample for each class in the training set
            train_class_indices = class_indices[:train_size]
            test_class_indices = class_indices[train_size:]
            
            # Append to the train and test lists
            train_idx[key].extend(train_class_indices)
            test_idx[key].extend(test_class_indices)
        #remove the validation index from the training index
        train_idx[key] = np.setdiff1d(train_idx[key], vald_idx_AUASE[key])
        # Shuffle the final training and testing indices to remove any class-based ordering
        np.random.shuffle(train_idx[key])
        np.random.shuffle(test_idx[key])

    # Convert lists back to arrays if needed
    for key in train_idx.keys():
        train_idx[key] = np.array(train_idx[key], dtype=int)
        test_idx[key] = np.array(test_idx[key], dtype=int)
    start_time = time.time()  # Start timing the iteration
    accuracies = []
    f1_scores = []
    f1_scores_micro = []
    f1_scores_macro = []
    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        X_train = YAcs[t][train_idx[key], :]
        X_test = YAcs[t][test_idx[key], :]
        y_train = labels_encoded[key][train_idx[key]]
        y_test = labels_encoded[key][test_idx[key]]
        label_encoder = LabelEncoder()

        # Fit on y_train so that classes are consistently mapped
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Number of unique classes
        k = len(label_encoder.classes_)


        model = xgb.XGBClassifier(objective='multi:softprob', num_class=k)

        # Training the model on the training data
        model.fit(X_train, y_train)

        # Making predictions on the test set
        predictions = model.predict(X_test)

        # Calculating accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        f1_scores.append(f1_score(y_test, predictions, average='weighted'))
        f1_scores_micro.append(f1_score(y_test, predictions, average='micro'))
        f1_scores_macro.append(f1_score(y_test, predictions, average='macro'))
        # print(f"Year {key}: Accuracy = {accuracy}, F1 Score = {f1_score(y_test, predictions, average='weighted')}")
    total_accuracies[i] = accuracies
    total_f1[i] = f1_scores
    total_f1_micro[i] = f1_scores_micro
    total_f1_macro[i] = f1_scores_macro

    print(f"Iteration {i}")

# %%
# do the mean over the 10 iterations
mean_accuracies = np.mean(total_accuracies, axis=0)
mean_f1 = np.mean(total_f1, axis=0)
mean_f1_micro = np.mean(total_f1_micro, axis=0)
mean_f1_macro = np.mean(total_f1_macro, axis=0)
#%%
# save the mean accuracies
np.save('mean_accuracies_CUASE.npy', mean_accuracies)
np.save('mean_f1_CUASE.npy', mean_f1)
#%%
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