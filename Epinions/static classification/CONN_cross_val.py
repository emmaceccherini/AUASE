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
import torch
import sklearn
np.random.seed(0)
#%%
T = 11
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
#%%
long_labels = np.hstack(list(labels_encoded.values()))
#%%
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
total_accuracies = np.zeros((len(alphas), len(labels_encoded.keys())))
vald_idx_DRLAN = {}
for a,alpha in enumerate(alphas):
    d = 22
    n = 15851
    embs = [torch.load(f'../CONN/embeddings_lp_Epinions_{t}_tw{alpha}.pth') for t in range(T)] 
    Y = np.vstack(embs)
    Y = Y.reshape(T, n, d)   
    np.random.seed(a)
    train_idx = {}
    val_idx = {}
 
    for key in labels_encoded.keys():
        # Get unique classes in the current label set
        unique_classes = np.unique(labels_encoded[key])
        
        # Create a list to hold training and testing indices
        train_idx[key] = []
        
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
            
            # Append to the train and test lists
            train_idx[key].extend(train_class_indices)
        # Shuffle the final training and testing indices to remove any class-based ordering
        np.random.shuffle(train_idx[key])
        perm_val = np.random.permutation(train_idx[key])
        val_idx[key] = perm_val[:len(perm_val)//10].astype(int)
        #remove the validation index from the training index
        train_idx[key] = np.setdiff1d(train_idx[key], val_idx[key])
    vald_idx_DRLAN[alpha] = val_idx
    # Convert lists back to arrays if needed
    for key in train_idx.keys():
        train_idx[key] = np.array(train_idx[key], dtype=int)

    accuracies = []

    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        X_train = Y[t][train_idx[key], :]
        X_val = Y[t][val_idx[key], :]

        y_train = labels_encoded[key][train_idx[key]]
        y_val = labels_encoded[key][val_idx[key]]
        
        label_encoder = LabelEncoder()

        # Fit on y_train so that classes are consistently mapped
        y_train = label_encoder.fit_transform(y_train)

        # Number of unique classes
        k = len(label_encoder.classes_)


        model = xgb.XGBClassifier(objective='multi:softprob', num_class=k)

        # Training the model on the training data
        model.fit(X_train, y_train)

        # Making predictions on the test set
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)
    total_accuracies[a] = accuracies
    print("Alpha: ", alpha)

# %%
alpha = alphas[np.argmax(np.mean(total_accuracies, axis =1))]
print(alpha)

#%%
# save the validation data
np.save('CONN_val_idx.npy', vald_idx_DRLAN[alpha])

# %%
