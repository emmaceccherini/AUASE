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
import sklearn
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
    d = 22
    X, Y = se.UASE(Acs, d)

    YAcs = np.zeros((T,n,d))
    for t in range(T):
        for i in range(n):
            if np.linalg.norm(Y[t,i]) > 1e-10:
                YAcs[t,i] = spherical_project(Y[t,i])
    
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
    vald_idx_AUASE[alpha] = val_idx
    # Convert lists back to arrays if needed
    for key in train_idx.keys():
        train_idx[key] = np.array(train_idx[key], dtype=int)

    accuracies = []

    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        X_train = YAcs[t][train_idx[key], :]
        X_val = YAcs[t][val_idx[key], :]

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
np.save('AUASE_val_idx.npy', vald_idx_AUASE[alpha])