#%%
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from scipy import io
from sklearn.preprocessing import LabelEncoder
np.random.seed(0)
#%%#%%

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
vald_idx_DRLAN = np.load('DRLAN_val_idx.npy', allow_pickle=True).item()
#%%
d = 22
T = 11
n = 15851
alpha = 0.5
embs = [io.loadmat(f'../DRLAN/embedding_{t+1}beta{alpha}.mat')['U'] for t in range(T)] 
Y = np.vstack(embs)
Y = Y.reshape(T, n, d) 
#%%
total_accuracies = np.empty((100, len(labels_encoded.keys())))
total_f1 = np.empty((100, len(labels_encoded.keys())))
total_f1_micro = np.empty((100, len(labels_encoded.keys())))
total_f1_macro = np.empty((100, len(labels_encoded.keys())))
iteration_time = []  # To store the time taken for each iteration
for i in range(100):
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

        train_idx[key] = np.setdiff1d(train_idx[key], vald_idx_DRLAN[key])
        # Shuffle the final training and testing indices to remove any class-based ordering
        np.random.shuffle(train_idx[key])
        np.random.shuffle(test_idx[key])

    # Convert lists back to arrays if needed
    for key in train_idx.keys():
        train_idx[key] = np.array(train_idx[key], dtype=int)
        test_idx[key] = np.array(test_idx[key], dtype=int)

    accuracies = []
    f1_scores = []
    f1_scores_micro = []
    f1_scores_macro = []
    for t, key in enumerate(labels_encoded.keys()):
        # XGBoost for classification
        X_train = Y[t][train_idx[key], :]
        X_test = Y[t][test_idx[key], :]
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
    

    print(f"Iteration {i} completed")

# %%
# do the mean over the 10 iterations
mean_accuracies = np.mean(total_accuracies, axis=0)
mean_f1 = np.mean(total_f1, axis=0)
mean_f1_micro = np.mean(total_f1_micro, axis=0)
mean_f1_macro = np.mean(total_f1_macro, axis=0)
#%%
# save the mean accuracies
np.save('mean_accuracies_DRLAN.npy', mean_accuracies)
np.save('mean_f1_DRLAN.npy', mean_f1)
#%%
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