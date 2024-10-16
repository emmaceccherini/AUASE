#%%
# load Rich_Epinions_Dataset_anonym/Trust.csv
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from scipy import io
#%%
userTrust = pd.read_csv('Epinions/dataset/userTrust.csv', header=None)
userTrusted = pd.read_csv('Epinions/dataset/userTrusted.csv', header=None)

# %%
combined_df = pd.concat([userTrust, userTrusted])
#%%
# remove rowa with column 2 equal to '-'
combined_df = combined_df[combined_df[2] != '-']
# %%
combined_df['Date'] = pd.to_datetime(combined_df[2], format='%b %d \'%y')
#%%
combined_df['Year'] = combined_df['Date'].dt.year
#%%
users = pd.unique(combined_df[[0, 1]].values.ravel('K'))
n = len(users)
#%%
# create a dictionary to map users to integers
user_dict = {user: i for i, user in enumerate(users)}
# %%
# Create adjacency matrix for each year
As = []
for year in np.sort(combined_df['Year'].unique()):
    year_data = combined_df[combined_df['Year'] == year]
    
    adj_matrix = lil_matrix((n, n), dtype=np.int32)
    
    # Fill in the adjacency matrix
    for _, row in year_data.iterrows():
        user_a = user_dict[row[0]]
        user_b = user_dict[row[1]]
        adj_matrix[user_a, user_b] = 1
        adj_matrix[user_b, user_a] = 1  # Since it's undirected
    
    # Display the adjacency matrix for this year
    print(f"\nAdjacency Matrix for the Year {year}:")
    #save the matrix to a file
    As.append(adj_matrix)

# %%
# only keep the users that have at least 3 interactions
# in teh whole time period
removed_user = []
removed_user_name = []
for i in range(n):
    sum = np.sum([A[i,: ] for A in As])
    if np.sum(np.sum(sum)) <2: # 2 
        # userlist.drop(i, inplace=True)
        removed_user.append(user_dict[users[i]])
        removed_user_name.append(users[i])
        # print(f"User {i} has been removed")
#%%
# get a list of non-removed users
active_users = [user for user in list(user_dict.values()) if user not in removed_user]
# %%
# remove the users from the adjacency matrices
As =[A[active_users, :][:, active_users] for A in As]

#%%
for i, A in enumerate(As):
    io.savemat(f'adj/adj_matrix_{i}.mat', {'A': A})
# %%
# remove the users from the user dictionary
for user in removed_user_name:
    del user_dict[user]
#%%
# reorder the user dictionary
user_dict = {user: i for i, user in enumerate(user_dict.keys())}
#%%
# save the dictionary to a file
np.save('user_dict.npy', user_dict)
# %%
print(As[0].shape)
print(len(user_dict))
# %%
