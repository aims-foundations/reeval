import pandas as pd
import re

# Load the CSV file
file_path = '../../data/real/crawl/crawl_dataset_name.csv'
df = pd.read_csv(file_path)

# Define a function to remove 'model=...' from the dataset information
def clean_dataset_name(name):
    return re.sub(r'model=[^,]*,?', '', name).strip(',')

# Apply the function to the 'Run' column to clean the dataset names
df['cleaned_run'] = df['Run'].apply(clean_dataset_name)

# Group by the cleaned dataset names and count the occurrences of each
grouped_df = df['cleaned_run'].value_counts().reset_index()
grouped_df.columns = ['dataset_info', 'model_count']

# Sort by model count in descending order
sorted_df = grouped_df.sort_values(by='model_count', ascending=False)

# Save the result to a new CSV file
output_file = '../../data/real/crawl/dataset_name_analysis.csv'
sorted_df.to_csv(output_file, index=False)
