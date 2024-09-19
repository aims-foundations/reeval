import numpy as np
import pandas as pd

def has_low_ratio(arr, ratio):
    arr_filtered = arr[arr != -1]
    if len(arr_filtered) == 0:
        return False
    counts = np.bincount(arr_filtered)
    max_ratio = np.max(counts) / len(arr_filtered)
    return max_ratio <= ratio

def filter_matrix(y_df_path, mask_save_path, unmask_save_path, ratio=0.8, strategy='colfirst'):
    y_df = pd.read_csv(y_df_path)
    row_names = y_df.iloc[:, 0].to_list()
    matrix = y_df.iloc[:, 1:].to_numpy()
    print(f'Original matrix shape: {matrix.shape}')

    if strategy == 'rowfirst':
        rows_to_keep = [i for i in range(matrix.shape[0]) if has_low_ratio(matrix[i, :], ratio)]
        filtered_matrix = matrix[rows_to_keep, :]
        cols_to_keep = [j for j in range(filtered_matrix.shape[1]) if has_low_ratio(filtered_matrix[:, j], ratio)]
        final_matrix = filtered_matrix[:, cols_to_keep]
    elif strategy == 'colfirst':
        cols_to_keep = [i for i in range(matrix.shape[1]) if has_low_ratio(matrix[:, i], ratio)]
        filtered_matrix = matrix[:, cols_to_keep]
        rows_to_keep = [j for j in range(filtered_matrix.shape[0]) if has_low_ratio(filtered_matrix[j, :], ratio)]
        final_matrix = filtered_matrix[rows_to_keep, :]
    print(f'Filtered matrix shape: {final_matrix.shape}')

    filtered_row_names = [row_names[i] for i in rows_to_keep]
    final_df = pd.DataFrame(final_matrix)
    final_df.columns = [f'{i}' for i in range(final_df.shape[1])]
    final_df.insert(0, '', filtered_row_names)
    final_df.to_csv(mask_save_path, index=False)
    
    unmask_matrix = final_matrix[:, ~np.any(final_matrix == -1, axis=0)]
    print(f'Unmask filtered matrix shape: {unmask_matrix.shape}')
    unmask_df = pd.DataFrame(unmask_matrix)
    unmask_df.columns = [f'{i}' for i in range(unmask_df.shape[1])]
    unmask_df.insert(0, '', filtered_row_names)
    unmask_df.to_csv(unmask_save_path, index=False)
    
if __name__ == "__main__":
    filter_matrix(
        y_df_path='../data/real/response_matrix/normal_syn_reason/mask_matrix_15k.csv',
        mask_save_path='../data/real/response_matrix/normal_syn_reason_clean/mask_matrix.csv',
        unmask_save_path='../data/real/response_matrix/normal_syn_reason_clean/non_mask_matrix.csv',
        ratio=0.9
    )
