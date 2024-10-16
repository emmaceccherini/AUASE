#%%
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
#%%
labels_encoded = np.load('labels_encoded.npy', allow_pickle=True).item()
#%%
labels = []
for key in labels_encoded.keys():
    labels.append(labels_encoded[key])
labels = np.concatenate(labels)
#%%
labels = labels[labels != 25]
#%%
plt.hist(labels, bins=25)
plt.show()
# %%
le = np.load('label_encoder.npy', allow_pickle=True).item()
# %%
# some labels are more popular than others
# so i combine them in a less number of labels
# print the total count of each label
print(np.unique(labels, return_counts=True)[1])
# %%
# Computer Hardware -> Computers & Internet
# Gift -> Online Stores & Services
# Media -> Movies 
# Photo & Optics -> Electronics
# Preview Categories -> Unlabelled
# Web Sites & Internet Services -> Computers & Internet
#%%
# change all the labels according to the above mapping
#%%
for key in labels_encoded.keys():
    labels_encoded[key][labels_encoded[key] == le.transform(['Computer Hardware'])[0]] = le.transform(['Computers & Internet'])[0]
    labels_encoded[key][labels_encoded[key] == le.transform(['Gifts'])[0]] = le.transform(['Online Stores & Services'])[0]
    labels_encoded[key][labels_encoded[key] == le.transform(['Media'])[0]] = le.transform(['Movies'])[0]
    labels_encoded[key][labels_encoded[key] == le.transform(['Photo & Optics'])[0]] = le.transform(['Electronics'])[0]
    labels_encoded[key][labels_encoded[key] == le.transform(['Preview Categories'])[0]] = le.transform(['Unlabelled'])[0]
    labels_encoded[key][labels_encoded[key] == le.transform(['Web Sites & Internet Services'])[0]] = le.transform(['Computers & Internet'])[0]
#%%
# load the user_dict
user_dict = pd.DataFrame(np.load('user_dict.npy', allow_pickle=True))
# reorder the column 1 to be equal to the index
user_dict[1] = range(0, len(user_dict))
#%%
remove_users = []
for i in range(len(user_dict)):
    labels = [labels_encoded[year][i] for year in labels_encoded.keys()]
    if np.all(np.array(labels) == 25):
        remove_users.append(i)
        # print(i)
# %%
# load the adjacency matrices
# load As
T = 11
# load adj_matrix_0.mat to adj_matrix_11.mat
As = [io.loadmat(f'adj/adj_matrix_{i}.mat')['A'].astype(np.float64) for i in range(T)]
#%%
# remove the users from the user_dict
user_dict = user_dict.drop(remove_users)
#%%
# remove the users from the adjacency matrices
active_users = list(user_dict[1])
As =[A[active_users, :][:, active_users] for A in As]

#%%

Cs = [io.loadmat(f'Cs/C_{2001+i}.mat')['C'].astype(np.float64) for i in range(T)]

#%%
# remove the users from the Cs
Cs = [C[active_users, :] for C in Cs]
#%%
# remove the users from the labels_encoded
labels_encoded = {key: labels_encoded[key][active_users] for key in labels_encoded.keys()}
# %%
# reorder user_dict column 1
user_dict[1] = range(0, len(user_dict))
#%%
# save the user_dict
np.save('user_dict.npy', user_dict)
#%%
# save the adjacency matrices
for i, A in enumerate(As):
    io.savemat(f'adj/adj_matrix_{i}.mat', {'A': A})
#%%
# save Cs
for i, C in enumerate(Cs):
    io.savemat(f'Cs/C_{i}.mat', {'C': sparse.csr_matrix(C)})
#%%
# save the labels_encoded
# remove the years before 2001
labels_encoded = {key: labels_encoded[key] for key in labels_encoded.keys() if int(key) >= 2001}
np.save('labels_encoded.npy', labels_encoded)
# %%
print(As[0].shape)
print(len(labels_encoded["2001"]))
print(Cs[0].shape)
print(len(user_dict))