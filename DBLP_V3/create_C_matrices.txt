#%% 
import pandas as pd
import glob
import pickle
import nltk
from nltk.corpus import stopwords   
from scipy import sparse
import re
import numpy as np
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
#%%
# Load the active authors
# load active authors
with open('active_authors.pkl', 'rb') as f:
    unique_authors = pickle.load(f)
n = len(unique_authors)
# %%
def is_normal_word(word):
    # Check if the word consists only of alphabetic characters or hyphenated words and is non-empty
    return re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", word) is not None

# Strip punctuation from the beginning and end of words
def strip_punctuation(word):
    return re.sub(r"^\W+|\W+$", "", word)

def to_lower(word):
    return word.lower()
#%%
def get_abstracts(df, unique_authors):
    # Create a dictionary with authors as keys and empty lists as values
    abstracts = {author: [] for author in unique_authors}
    # Iterate over the DataFrame only once
    for i, row in df.iterrows():
        authors_in_row = row['Authors']
        abstract = row['Abstract']
        title = row['Title']

        for author in unique_authors:
            if author in authors_in_row:
                words = abstract.split()
                words.extend(title.split())
                words = [strip_punctuation(word) for word in words]
                words = [word for word in words if is_normal_word(word)]
                words = [to_lower(word) for word in words]
                abstracts[author].extend(words)
    return abstracts
# %%
abstracts = {}
for k,v in dfs.items():
    abstract_t = get_abstracts(v, unique_authors)
    abstracts[k] = abstract_t

# %%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


from collections import defaultdict, Counter
all_words = Counter()
for k,v in abstracts.items():
    for author_id, words_dic in v.items():
        words = defaultdict(list)
        for word in words_dic:
            words[word].extend([1])
        words_update ={key: len(value) for key, value in words.items()}
        all_words.update(words_update)
# %%
all_words_counter = dict(all_words)
all_words = list(all_words.keys())
#%%
l = np.array([len(word) for word in all_words_counter.keys() ])
# remove them from the dictionary 
for word in np.array(all_words)[np.where(l <3)[0]]:
    all_words_counter.pop(word)


# %%
# remove the words that have count less than 100
all_words_counter = {key: value for key, value in all_words_counter.items() if value >= 100}
# remove the words that have count higher than 12000
all_words_counter = {key: value for key, value in all_words_counter.items() if value <= 5000}
#%%
word_to_index = {word: idx for idx, word in enumerate(all_words_counter)}

#%%
with open("word_to_index.pkl", 'wb') as file:
    pickle.dump(word_to_index, file)
# %%
author_indices = {author: index for index, author in enumerate(unique_authors)}
# %%
num_authors = len(author_indices)
num_words = len(word_to_index)
#%% 
for k,v in dfs.items():
    author_ids = list(abstracts[k].keys())
    row_indices = []
    col_indices = []
    values = []
    abstract = abstracts[k]
    for author_id in author_ids:
        # get the index of the author
        words = abstract[author_id]
        for word in words:
            if word not in word_to_index:
                continue
            word_idx = word_to_index[word]
            row_indices.append(author_indices[author_id])
            col_indices.append(word_idx)
            values.append(1)
    # Create a sparse matrix
    word_count_matrix = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(num_authors, num_words))

    # Converting to CSR format for efficient row slicing
    word_count_matrix = word_count_matrix.tocsr()

    sparse.save_npz(f'C_matrices/word_count_matrix_{k}.npz', word_count_matrix)



