#%%
import numpy as np
import pandas as pd
import os
from scipy import io
from collections import defaultdict, Counter
import re
import nltk
from nltk.corpus import stopwords
import pickle
#%%
# load user_dict.npy
user_dict = np.load('user_dict.npy', allow_pickle=True).item()
# %%
dfs = {}

# Iterate over the dictionary
for user, idx in user_dict.items():
    # Generate the file name
    file_name = f"Epinions/dataset/reviews/{user}_reviews.csv"
    
    # Check if the file exists (this is important to avoid errors)
    if os.path.exists(file_name):
        # if the file is empty, skip it
        if os.stat(file_name).st_size == 0:
            continue
        # Load the CSV into a DataFrame
        df = pd.read_csv(file_name)
        if len(df) == 0:
            continue
        # add to each df the column names
        df.columns = ['date', 'title', 'link', 'item', 'link2', 'subcategory', 'link3', 'category', 'rating', 'help']
        # only keep the columnd date tile and category
        df = df[['date', 'title', 'category']]
        dfs[user] = df
# %%
# load adj/adj_matrix_1.mat
As= []
for i in range(11):
    As.append(io.loadmat(f'adj/adj_matrix_{i}.mat')['A'])
# %%
active_users = list(dfs.keys())
active_users_num = [user_dict[user] for user in active_users]
#%%
removed_user = [user for user in list(user_dict.keys()) if user not in active_users]
# %%
# remove the users from the adjacency matrices
As =[A[active_users_num, :][:, active_users_num] for A in As]

#%%
for i, A in enumerate(As):
    io.savemat(f'adj/adj_matrix_{i}.mat', {'A': A})
# %%

# remove the users from the user dictionary
for user in removed_user:
    del user_dict[user]
#%%
# reorder the user dictionary
user_dict = {user: i for i, user in enumerate(user_dict.keys())}
#%%
# save the dictionary to a file
np.save('user_dict.npy', user_dict)

# %%
nltk.download('stopwords')

# Get the list of English stop words
stop_words = set(stopwords.words('english'))

all_words = Counter()

# Tokenize and count words in each user's review titles
for user, df in dfs.items():
    # Iterate over each review title in the user's DataFrame
    for title in df['title']:
        # Convert title to lowercase, remove punctuation, and split into words
        words = re.findall(r'\w+', title.lower())  # This removes punctuation and tokenizes words
        filtered_words = [word for word in words if word not in stop_words]
        # Update the Counter with words from this title
        all_words.update(filtered_words)
#%%
all_words_counter = dict(all_words)
all_words = list(all_words.keys())

#%%
l = np.array([len(word) for word in all_words_counter.keys() ])
# remove them from the dictionary 
for word in np.array(all_words)[np.where(l <3)[0]]:
    all_words_counter.pop(word)
    all_words.remove(word)

#%%
# remove numbers
for word in all_words:
    if word.isnumeric():
        all_words_counter.pop(word)
#%%
# remove the words that have count less than 100
all_words_counter = {key: value for key, value in all_words_counter.items() if value >= 100}
# remove the words that have count higher than 12000
all_words_counter = {key: value for key, value in all_words_counter.items() if value <= 10000}
# %%
#%%
word_to_index = {word: idx for idx, word in enumerate(all_words_counter)}

#%%
with open("word_to_index.pkl", 'wb') as file:
    pickle.dump(word_to_index, file)
# %%
n_users = len(user_dict)
n_words = len(word_to_index)
#%% 
Cs = defaultdict(lambda: np.zeros((n_users, n_words), dtype=int))
#%%
# Loop through each user and their reviews
for user, df in dfs.items():
    user_idx = user_dict[user]
    df['date'] = pd.to_datetime(df["date"], format='%b %d \'%y')
    df['year'] = df['date'].dt.year
    # Loop through each review
    for _, row in df.iterrows():
        title = row['title']
        year = row['year']
        
        # Tokenize the review title
        words = re.findall(r'\w+', title.lower())
        
        # Count word occurrences in the title
        word_counts = Counter(words)
        
        # Update the matrix for the given year
        for word, count in word_counts.items():
            if word in word_to_index:
                word_idx = word_to_index[word]
                Cs[year][user_idx, word_idx] += count
    # print(f"Finished processing reviews for user {user}")
#%%
for year in range(2001, 2012):
    io.savemat(f'Cs/C_{year}.mat', {'C': Cs[year]})
    print(f"Saved matrix for year {year}")
# %%
print(As[0].shape)
print(len(user_dict))
print(Cs[2001].shape)
print(len(Cs))