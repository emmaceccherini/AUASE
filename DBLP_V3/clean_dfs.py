#%%
import pandas as pd
import numpy as np
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
df = parse_data("DBLP-citation-network-Oct-19/DBLPOnlyCitationOct19.txt")

#%%
# remove all the rows qith year 
# equal to -1, '1945', '1954'
df["Year"] = df["Year"].astype(int)

df = df[~df['Year'].isin(['-1'])]
#%%
# # remove years before 1985
df = df[df['Year'] > 1995]
df = df[df['Year'] < 2010]
#%%
# %%
df['Authors'] = df['Authors'].str.split(',')
# %%
# remove all the entries with only one author
df = df[df['Authors'].apply(lambda x: len(x) > 1)]
# %%
# remove all the rows whose pubblication venue is not in the list of labels
#%%
# get the list of pubblcation venues
unique_venues = list(df['Publication Venue'].unique())

#%%
df.loc[df['Publication Venue'].str.startswith('HCI'), 'Publication Venue'] = 'HCI'

#%%
# only keep rows for which the publication venue is in this list 
# "PPOPP"  "DAC" "MICRO" "PODC" "SIGCOMM" "MOBICOM" "INFOCOM" 
# "SenSys" "SIGMOD Conference" "ICDE" "SIGIR" "STOC"
# "SODA" "CAV" "FOCS" "SIGGRAPH" "IEEE Visualization" 
# "ICASSP" "IJCAI" "ACL" "NIPS" "IUI" "PerCom" "HCI"

venues = ["PPOPP", "PPOPP/PPEALS", "ASP-DAC", "EURO-DAC", "DAC",
           "MICRO","DIALM-PODC", "PODC", "SIGCOMM", "MOBICOM",
         "MOBICOM-CoRoNet", "INFOCOM", "SIGMOD Workshop, Vol. 1",
         "SIGMOD Workshop, Vol. 2", "SIGSMALL/SIGMOD Symposium",
         "SIGMOD Record", "ICDE Workshops", "SIGIR Forum", 
         "IJCAI (1)", "IJCAI (2)", "ACL2", "PerCom Workshops"
          "SenSys", "SIGMOD Conference", "ICDE", "SIGIR", "STOC",
          "SODA", "CAV", "FOCS", "SIGGRAPH", "IEEE Visualization", 
          "ICASSP", "IJCAI", "ACL", "NIPS", "IUI", "PerCom", "HCI"]

df = df[df['Publication Venue'].isin(venues)]
#%%
df.loc[df['Publication Venue'].isin(["PPOPP", "PPOPP/PPEALS", "ASP-DAC", 
                                     "EURO-DAC", "DAC", "MICRO", "PODC", "DIALM-PODC"]), 'Label'] = "Computer Architecture"
df.loc[df['Publication Venue'].isin(["SIGCOMM", "MOBICOM", "MOBICOM-CoRoNet", "INFOCOM", "SenSys"]), 'Label'] = "Computer Network"
df.loc[df['Publication Venue'].isin(["SIGMOD Conference","SIGMOD Workshop, Vol. 1",
         "SIGMOD Workshop, Vol. 2", "SIGSMALL/SIGMOD Symposium",
         "SIGMOD Record", "ICDE", "ICDE Workshops", "SIGIR", "SIGIR Forum"]), 'Label'] = "Data Mining"
df.loc[df['Publication Venue'].isin(["STOC", "SODA", "CAV", "FOCS"]), 'Label'] = "Computer Theory"
df.loc[df['Publication Venue'].isin(["SIGGRAPH", "IEEE Visualization", "ICASSP"]), 'Label'] = "Multi-Media"
df.loc[df['Publication Venue'].isin(["IJCAI","IJCAI (1)", "IJCAI (2)","ACL2",  "ACL", "NIPS"]), 'Label'] = "Artificial Intelligence"
df.loc[df['Publication Venue'].isin(["IUI", "PerCom", "HCI"]), 'Label'] = "Computer-Human Interaction"

# %%
# reorder the columns 
df = df.reset_index(drop=True)

# %%
dfs = {year: df[df['Year'] == year] for year in df['Year'].unique()}
# %%
# print the lenght of each dataframe
lengths = {year: len(dfs[year]) for year in np.unique(df['Year'])}

#%%

def map_year_to_interval(year):
    if year >= 2003:
        return year
    elif year in [2001, 2002]:
        return "2001 - 2002"
    elif year in [1996, 1997, 1998, 1999, 2000]:
        return "1996 - 2000"
    elif year in [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995]:
        return "1986 - 1995"
    else:
        return "1960 - 1985"

# %%
df['Interval'] = df['Year'].apply(map_year_to_interval)
df = df.reset_index(drop=True)

# %%
dfs = {interval: group for interval, group in df.groupby('Interval')}
# %%
lengths = {int: len(dfs[int]) for int in dfs.keys()}
#%%
dfs = {key: dfs[key].reset_index(drop=True) for key in dfs.keys()}
#%%
# Count the number of papers each author has published
author_counts = pd.Series([author for sublist in df['Authors'].tolist() for author in sublist]).value_counts()

# Create a list of authors who have published less than 3 papers
inactive_authors = author_counts[author_counts < 2].index.tolist()

#%%
# get the list of authors 
unique_authors = pd.Series([author for sublist in df['Authors'].tolist() for author in sublist]).unique()
#%%
# Convert unique_authors and inactive_authors to sets
active_authors = [author for author in sorted(unique_authors) if author not in inactive_authors]

# %%
import pickle

with open('active_authors.pkl', 'wb') as f:
    pickle.dump(active_authors, f)
# %%
for interval, df_interval in dfs.items():
    df_interval.to_csv(f'dfs/df_{interval}.csv', index=False)
# %%
# save the active authors
with open('active_authors.pkl', 'wb') as f:
    pickle.dump(active_authors, f)

