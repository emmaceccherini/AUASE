#%%
import glob
import pandas as pd
import numpy as np
from scipy import io
# %%
# Find all the files that start with 'labels' and end with '.pkl'
files = glob.glob('labels/user_labels*.txt')

# Load each file
labels = {}
for file in files:
    labels[file] = pd.read_csv(file, sep=':', header=None, names=['user', 'label'])
    
# %%
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
# load user_dict.npy
user_dict = np.load('user_dict.npy', allow_pickle=True).item() 
user_dict = pd.DataFrame.from_dict(user_dict, orient='index').reset_index()
user_dict.columns = ['user', 'index']
#%%
reordered_labels = {}

# Iterate through each year in the labels dictionary
for year, df in labels.items():
    df['label'] = df['label'].str.strip()
    # Merge the current year's dataframe with the user_dict
    merged_df = pd.merge(user_dict, df, on='user', how='left')
    
    # Fill missing labels with 'Unlabelled'
    merged_df['label'].fillna('Unlabelled', inplace=True)
    
    # Reorder the dataframe based on the 'index' column
    merged_df = merged_df.sort_values(by='index').reset_index(drop=True)
    
    # Drop the 'index' column if you no longer need it
    merged_df = merged_df.drop(columns=['index'])
    
    # Store the reordered dataframe back into the dictionary
    reordered_labels[year] = merged_df
#%%
# check that a user has a label different from 'Unlabelled'
# for at least one year
remove_users = []
for user in user_dict['user']:
    labels = [reordered_labels[year][reordered_labels[year]['user'] == user]['label'].values[0] for year in reordered_labels.keys()]
    if np.all(np.array(labels) == 'Unlabelled'):
        remove_users.append(user)
        print(user)
#%% 
# these users only ahev archive reviews
# remove them from the user_dict
user_dict = user_dict[~user_dict['user'].isin(remove_users)]
#%%
# save the user_dict
np.save('user_dict.npy', user_dict)
#%%
 #%%
from sklearn.preprocessing import LabelEncoder

# Get all unique labels across all time points
all_labels = set()
for values in reordered_labels.values():
    all_labels.update(values["label"])
all_labels = list(all_labels)
all_labels.remove('<a href=/member>Member Center</a>')
all_labels.remove('<font color=666666><i>Category information unavailable.</i></font>')

# Fit the LabelEncoder on all unique labels
le = LabelEncoder()
le.fit(list(all_labels))
# remove the first 2 labels
# save the label encoder
np.save('label_encoder.npy', le)

#%%
# Transform the labels at each time point
labels_encoded = {}
for key, values in reordered_labels.items():
    # if the category is <a href=/member>Member Center</a> replace it with 'Unlabelled'
    values['label'] = values['label'].replace('<a href=/member>Member Center</a>', 'Unlabelled')
    values['label'] = values['label'].replace('<font color=666666><i>Category information unavailable.</i></font>', 'Unlabelled')
    labels_encoded[key] = le.transform(values["label"])
# %%
# save the labels_encoded
np.save('labels_encoded.npy', labels_encoded)
# Unalbelled is 25

# %%
# load As
As = [io.loadmat(f'adj/adj_matrix_{i}.mat')['A'] for i in range(11)]
Cs = [io.loadmat(f'Cs/C_{2001+i}.mat')['C'] for i in range(11)]
#%%
# remove the users from the adjacency matrices
As =[A[user_dict['index'], :][:, user_dict['index']] for A in As]
Cs =[C[user_dict['index'], :] for C in Cs]
#%%
# reorder user dictionary
user_dict[1] = np.arange(len(user_dict))
#%%
# save Cs
for i, C in enumerate(Cs):
    io.savemat(f'Cs/C_{2001+i}.mat', {'C': C})

for i, A in enumerate(As):
    io.savemat(f'adj/adj_matrix_{i}.mat', {'A': A})
# %%
# save the dictionary to a file
np.save('user_dict.npy', user_dict)
#%%
# remove the users from the labels_encoded
for key in labels_encoded.keys():
    labels_encoded[key] = labels_encoded[key][user_dict['index']]
#%%
# save the labels_encoded
np.save('labels_encoded.npy', labels_encoded)
# %%
print(As[0].shape)
print(len(labels_encoded["2001"]))
print(Cs[0].shape)
print(len(user_dict))
#%%
