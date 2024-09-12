import pandas as pd
import re

def clean_dataset_name(name):
    return re.sub(r'model=[^,]*,?', '', name).strip(',')
    
if __name__ == "__main__":
    exp = "lite"
    
    file_path = f'../../data/real/crawl/crawl_dataset_name_{exp}.csv'
    df = pd.read_csv(file_path)

    df['cleaned_run'] = df['Run'].apply(clean_dataset_name)
    grouped_df = df['cleaned_run'].value_counts().reset_index()
    grouped_df.columns = ['dataset_info', 'model_count']

    sorted_df = grouped_df.sort_values(by='model_count', ascending=False)

    output_file = f'../../data/real/crawl/dataset_name_analysis_{exp}.csv'
    sorted_df.to_csv(output_file, index=False)

