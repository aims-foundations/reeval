import pandas as pd
import os

if __name__ == "__main__":
    input_dir = "../../../data/real/pre_irt_data/eval"
    perturb_list = ["base", "perturb1", "perturb2"]
    i = 0

    for perturb in perturb_list:
        matrix_df = pd.DataFrame()

        for filename in os.listdir(input_dir):
            if filename.endswith(".csv"):
                infile_path = os.path.join(input_dir, filename)
                data = pd.read_csv(infile_path)
                columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
                data = data[columns_to_keep]
                
                # Select every 3rd row
                data = data.iloc[i::3, :]

                # Extract the last column which is the score
                last_column = data.iloc[:, -1]
                model_name = filename.split(f"eval_")[1].split("_result.csv")[0]
                matrix_df[model_name] = last_column

        matrix_df = matrix_df.replace(0.5, 1)
        matrix_df = matrix_df.astype(int)
        
        # delete questions with the same answers
        for index, row in matrix_df.iterrows():
            if row.nunique() == 1:
                matrix_df = matrix_df.drop(index, axis=0)

        matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)
        matrix_df.reset_index(drop=True, inplace=True)
        matrix_df = matrix_df.T
        
        matrix_df.to_csv(f'../../../data/real/response_matrix/{perturb}_matrix.csv')
        i += 1



    # get response matrix for all questions
    matrix_df = pd.DataFrame()
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir, filename)
            data = pd.read_csv(infile_path)
            columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
            data = data[columns_to_keep]
            last_column = data.iloc[:, -1]
            model_name = filename.split(f"eval_")[1].split("_result.csv")[0]
            matrix_df[model_name] = last_column
            
    matrix_df = matrix_df.replace(0.5, 1)
    matrix_df = matrix_df.astype(int)
    bool_delete_list = []
    for index, row in matrix_df.iterrows():
        if row.nunique() == 1:
            matrix_df = matrix_df.drop(index, axis=0)
            bool_delete_list.append(1)
        else:
            bool_delete_list.append(0)
    matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)
    matrix_df.reset_index(drop=True, inplace=True)
    matrix_df = matrix_df.T
    matrix_df.to_csv(f'../../../data/real/response_matrix/all_matrix.csv')
    
    
    
    # index search file
    input_file = "/Users/tyhhh/Desktop/certified-eval/data/real/pre_irt_data/eval/eval_01-ai_yi-34b-chat_result.csv"
    output_file = "../../../data/real/response_matrix/insex_search.csv"

    data = pd.read_csv(input_file)
    output_data = []

    for idx, row in data.iterrows():
        text = row.iloc[4]
        perturb = 'base' if idx % 3 == 0 else 'perturb1' if idx % 3 == 1 else 'perturb2'
        output_data.append([idx, text, perturb, bool_delete_list[idx]])

    output_df = pd.DataFrame(output_data, columns=["idx", "text", "perturb", "is_deleted"])
    output_df.to_csv(output_file, index=False)