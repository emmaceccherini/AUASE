#%% 
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import numba as nb
from scipy import io
import glob
import pickle
import ast
import statistics
#%%
# Find all CSV files in the current directory that start with 'df'
csv_files = glob.glob('dfs/df*.csv')

# Load each file into a DataFrame
dfs = {file: pd.read_csv(file) for file in csv_files}
#%%
# change the keys to the interval
new_dfs = {}
for key in list(dfs.keys()):
    interval = key.split('_')[-1].split('.')[0]
    new_dfs[interval] = dfs[key]
dfs = new_dfs

def extract_first_year(key):
    return int(key.split(' - ')[0])

# Sort the keys and reorder the dictionary
sorted_keys = sorted(dfs.keys(), key=extract_first_year)
dfs = {key: dfs[key] for key in sorted_keys}
# %%
# get the list of authors for each interval 
authors_list = {}
for interval, df in dfs.items():
    authors_list[interval] = df['Authors']

# %%
# load active authors
with open('active_authors.pkl', 'rb') as f:
    unique_authors = pickle.load(f)
n = len(unique_authors)

#%%
from collections import defaultdict
import statistics

def get_labels(df, unique_authors):
    # Create a dictionary where the keys are the authors and the values are the labels of their papers
    author_labels = defaultdict(list)
    for authors, label in zip(df["Authors"].apply(ast.literal_eval), df["Label"]):
        for author in authors:
            if author in unique_authors:
                author_labels[author].append(label)

    # Find the mode of the labels for each author
    labels = {author: statistics.mode(labels) for author, labels in author_labels.items() if labels}

    return labels
# %%
for interval, df in dfs.items():
    labels = get_labels(df, unique_authors)
    author_labels = [labels.get(author, 'unlabeled') for author in unique_authors]
    with open(f'labels/labels_{interval}.pkl', 'wb') as f:
        pickle.dump(author_labels, f)
# %%
