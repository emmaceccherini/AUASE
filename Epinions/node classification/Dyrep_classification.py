import numpy as np
import pickle
from scipy import io
import xgboost as xgb
import glob
from sklearn.metrics import accuracy_score,f1_score 
#%%
d = 22
T = 11
n = 15851

embs = [np.load(f'../DyRep/embeddings_before_time_{t+1}_Epinions.npy')[:n,] for t in range(T)]

Y = np.vstack(embs)
Y = Y.reshape(T, n, d)   
#%%

# %%
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
active = np.empty((T,n), dtype=bool)
for t in range(T):
    key = list(labels_encoded.keys())[t]
    active[t] = labels_encoded[key] !=21

#%%
X_train = np.vstack(Y[:6])
X_train = X_train[active[:6].reshape(-1)]
y_train = np.hstack(list(labels_encoded.values())[:6])
y_train = y_train[active[:6].reshape(-1)]

X_test = np.vstack(Y[6:])
X_test = X_test[active[6:].reshape(-1)]
y_test = np.hstack(list(labels_encoded.values())[6:])
y_test = y_test[active[6:].reshape(-1)]
#%%
model = xgb.XGBClassifier(objective='multi:softprob',
        num_class=21)

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test )
#%%
accuracies = []
f1_scores = []
f1_scores_micro = []
f1_scores_macro = []
#Calculating accuracy
n1 = active[6].sum() 
n2 = n1 + active[7].sum()
n3 = n2 + active[8].sum()
n4 = n3 + active[9].sum()


accuracies.append(accuracy_score(y_test[:n1], predictions[:n1]))
accuracies.append(accuracy_score(y_test[n1:n2], predictions[n1:n2]))
accuracies.append(accuracy_score(y_test[n2:n3], predictions[n2:n3]))
accuracies.append(accuracy_score(y_test[n3:n4], predictions[n3:n4]))
accuracies.append(accuracy_score(y_test[n4:], predictions[n4:]))


f1_scores.append(f1_score(y_test[:n1], predictions[:n1], average='weighted'))
f1_scores.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='weighted'))
f1_scores.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='weighted'))
f1_scores.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='weighted'))
f1_scores.append(f1_score(y_test[n4:], predictions[n4:], average='weighted'))

f1_scores_micro.append(f1_score(y_test[:n1], predictions[:n1], average='micro'))
f1_scores_micro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='micro'))
f1_scores_micro.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='micro'))
f1_scores_micro.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='micro'))
f1_scores_micro.append(f1_score(y_test[n4:], predictions[n4:], average='micro'))

f1_scores_macro.append(f1_score(y_test[:n1], predictions[:n1], average='macro'))
f1_scores_macro.append(f1_score(y_test[n1:n2], predictions[n1:n2], average='macro'))
f1_scores_macro.append(f1_score(y_test[n2:n3], predictions[n2:n3], average='macro'))
f1_scores_macro.append(f1_score(y_test[n3:n4], predictions[n3:n4], average='macro'))
f1_scores_macro.append(f1_score(y_test[n4:], predictions[n4:], average='macro'))
#%%
#%%
# save the accuracies 
with open('accuracies_Dyrep_temp.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
#%%
# save the f1 scores
with open('f1_scores_Dyrep_temp.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
# %%
print(np.round(accuracies, 3))
print(np.round(f1_scores, 3))
print(np.round(f1_scores_micro, 3))
print(np.round(f1_scores_macro, 3))

# %%
print(np.round(np.mean(accuracies), 3))
# %%
