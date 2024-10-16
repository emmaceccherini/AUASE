#%%
import numpy as np
import pandas as pd
import os
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
        df = pd.read_csv(file_name, header=None)
        if len(df) == 0:
            continue
        # add to each df the column names
        df.columns = ['date', 'title', 'link', 'item', 'link2', 'subcategory', 'link3', 'category', 'rating', 'help']
        # only keep the columnd date tile and category
        df = df[['date', 'title', 'category']]
        dfs[user] = df
# %%
user_majority_categories = {}

# Loop through each user and their DataFrame
for user, df in dfs.items():
    # Replace any missing categories (None) with 'Unlabelled'
    df['date'] = pd.to_datetime(df['date'],  format='%b %d \'%y')
    df['year'] = df['date'].dt.year
    
    # Group by year and category, then count the number of reviews per category in each year
    # ignore the ' ' category
    majority_category_per_year = df[df['category'] != ' '].groupby(['year', 'category']).size().reset_index(name='counts')
    # majority_category_per_year = df.groupby(['year', 'category']).size().reset_index(name='counts')
    
    # Find the majority category for each year
    for year in majority_category_per_year['year'].unique():
        year_data = majority_category_per_year[majority_category_per_year['year'] == year]
        
        # Find the category with the maximum count for this year
        majority_category = year_data.loc[year_data['counts'].idxmax(), 'category']
        
        # Store the result in a dictionary for this user
        if user not in user_majority_categories:
            user_majority_categories[user] = {}
        if majority_category == ' ':
            majority_category = 'Unlabelled'
        user_majority_categories[user][year] = majority_category

# %%
# Dictionary to group users by year
yearly_labels = {}

# Loop through the user-majority categories to organize data by year
for user, categories_per_year in user_majority_categories.items():
    for year, category in categories_per_year.items():
        if year not in yearly_labels:
            yearly_labels[year] = []
        # Append user: label for each year
        yearly_labels[year].append(f"{user}: {category}")
#%%
# Save each year's results to a separate file
for year, user_labels in yearly_labels.items():
    # Create a filename for the year
    file_name = f"labels/user_labels_{year}.txt"
    
    # Write the user: label pairs to the file
    with open(file_name, 'w') as file:
        for label in user_labels:
            file.write(f"{label}\n")
    
    print(f"Saved {file_name}")


# %%
