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
years = range(2000, 2015)
dfs = {}
for year in years:
    dfs[year] = pd.read_csv(f'dfs/df_{year}.csv')

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
            author = author.strip()
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


