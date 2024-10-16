#%%
import pandas as pd
import numpy as np
#%%
# citation-acm-v8.txt in the same directory 
#%%
def parse_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Split the data into individual entries
    entries = data.strip().split("\n\n")
    
    # Initialize a list to hold parsed entries
    parsed_entries = []

    # Parse each entry
    for entry in entries:
        entry_data = {
            'Title': None,
            'Authors': None,
            'Year': None,
            'Publication Venue': None,
            'Abstract': None
        }
        lines = entry.split('\n')
        
        for line in lines:
            if line.startswith('#*'):
                entry_data['Title'] = line[2:].strip()
            elif line.startswith('#@'):
                entry_data['Authors'] = line[2:].strip()
            elif line.startswith('#t'):
                entry_data['Year'] = line[2:].strip()
            elif line.startswith('#c'):
                entry_data['Publication Venue'] = line[2:].strip()
            elif line.startswith('#!'):
                entry_data['Abstract'] = line[2:].strip()
        
        # Only add entries that have all the required fields
        required_fields = ['Title', 'Authors', 
                           'Year', 'Publication Venue', "Abstract"]
        if all(entry_data.get(field) for field in required_fields):
            parsed_entries.append(entry_data)
    # Convert parsed entries to a DataFrame
    df = pd.DataFrame(parsed_entries)
    return df
#%%
df = parse_data("citation-acm-v8.txt")

# %%
df["Year"] = df["Year"].astype(int)
# %%
df['Authors'] = df['Authors'].str.split(',')
# %%
# remove all the entries with only one author
df = df[df['Authors'].apply(lambda x: len(x) > 1)]
#%%
# get the list of pubblcation venues
unique_venues = list(df['Publication Venue'].unique())
# save the list of publication venues
with open('venues.txt', 'w') as f:
    for item in unique_venues:
        f.write("%s\n" % item)
# %%
#reorder the rows of the dataframe
df = df.reset_index(drop=True)

# %%
# selection of the publication venues

df.loc[df['Publication Venue'].str.contains('VLDB', na=False), 'Publication Venue'] = 'VLDB'
df.loc[df['Publication Venue'].str.contains('SIGMOD', na=False), 'Publication Venue'] = 'SIGMOD'
df.loc[df['Publication Venue'].str.contains('PODS', na=False), 'Publication Venue'] = 'PODS'
df.loc[df['Publication Venue'].str.contains('ICDE', na=False), 'Publication Venue'] = 'ICDE'
df.loc[df['Publication Venue'].str.contains('EDBT', na=False), 'Publication Venue'] = 'EDBT'
df.loc[df['Publication Venue'].str.contains('SIGKDD', na=False), 'Publication Venue'] = 'SIGKDD'
df.loc[df['Publication Venue'].str.contains('ICDM', na=False), 'Publication Venue'] = 'ICDM'
df.loc[df['Publication Venue'].str.contains('DASFAA', na=False), 'Publication Venue'] = 'DASFAA'
df.loc[df['Publication Venue'].str.contains('SSDBM', na=False), 'Publication Venue'] = 'SSDBM'
df.loc[df['Publication Venue'].str.contains('CIKM', na=False), 'Publication Venue'] = 'CIKM'
df.loc[df['Publication Venue'].str.contains('PAKDD', na=False), 'Publication Venue'] = 'PAKDD'
df.loc[df['Publication Venue'].str.contains('PKDD', na=False), 'Publication Venue'] = 'PKDD'
df.loc[df['Publication Venue'].str.contains('SDM', na=False), 'Publication Venue'] = 'SDM'
df.loc[df['Publication Venue'].str.contains('DEXA', na=False), 'Publication Venue'] = 'DEXA'
df.loc[df['Publication Venue'].str.contains('CVPR', na=False), 'Publication Venue'] = 'CVPR'
df.loc[df['Publication Venue'].str.contains('ICCV', na=False), 'Publication Venue'] = 'ICCV'
df.loc[df['Publication Venue'].str.contains('ICIP', na=False), 'Publication Venue'] = 'ICIP'
df.loc[df['Publication Venue'].str.contains('ICPR', na=False), 'Publication Venue'] = 'ICPR'
df.loc[df['Publication Venue'].str.contains('ECCV', na=False), 'Publication Venue'] = 'ECCV'
df.loc[df['Publication Venue'].str.contains('ICME', na=False), 'Publication Venue'] = 'ICME'
df.loc[df['Publication Venue'].str.contains('ACM-MM', na=False), 'Publication Venue'] = 'ACM-MM'
#%%
venues = ['VLDB', 'SIGMOD', 'PODS', 'ICDE', 'EDBT', 'SIGKDD', 'ICDM', 
          'DASFAA', 'SSDBM', 'CIKM', 'PAKDD', 'PKDD', 'SDM', 'DEXA', 
          'CVPR', 'ICCV', 'ICIP', 'ICPR', 'ECCV', 'ICME', 'ACM-MM']

df = df[df['Publication Venue'].isin(venues)]
# %%
df.loc[df['Publication Venue'].isin(['VLDB', 'SIGMOD', 'PODS', 'ICDE', 'EDBT', 'SIGKDD', 'ICDM', 
          'DASFAA', 'SSDBM', 'CIKM', 'PAKDD', 'PKDD', 'SDM', 'DEXA']), 'Label'] = "Data Science"
df.loc[df['Publication Venue'].isin(['CVPR', 'ICCV', 'ICIP', 'ICPR', 'ECCV', 'ICME', 'ACM-MM']), 'Label'] = "Computer Vision"
# %%
df = df.reset_index(drop=True)
#%%

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 7))
sns.histplot(df['Year'], bins=45)
plt.title('Number of papers published each year')
plt.xlabel('Year')
#%%
# only consider years 2000-2014
df = df[df['Year'] < 2015]
df = df[df['Year'] > 1999]
#%%
# get the list of authors 
unique_authors = pd.Series([author for sublist in df['Authors'].tolist() for author in sublist]).unique()

unique_authors = np.array([name.strip() for name in unique_authors])
# %%
# Count the number of papers each author has published
author_counts = pd.Series([author for sublist in df['Authors'].tolist() for author in sublist]).value_counts()

# Create a list of authors who have published less than 3 papers
inactive_authors = author_counts[author_counts < 2].index.tolist()

# %%
# Convert unique_authors and inactive_authors to sets
unique_authors_set = set(unique_authors)
inactive_authors_set = set(inactive_authors)

# Subtract inactive_authors_set from unique_authors_set to get active_authors
active_authors = list(unique_authors_set - inactive_authors_set)
# %%
import pickle

with open('active_authors.pkl', 'wb') as f:
    pickle.dump(active_authors, f)

#%%
dfs = {year: df[df['Year'] == year] for year in df['Year'].unique()}
# %%
# print the lenght of each dataframe
lengths = {year: len(dfs[year]) for year in np.unique(df['Year'])}

#%%
for year, df in dfs.items():
    df.to_csv(f'dfs/df_{year}.csv', index=False)
# %%
# save the active authors
with open('active_authors.pkl', 'wb') as f:
    pickle.dump(active_authors, f)
# %%