import numpy as np
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = load_dataset("stair-lab/airbench-difficulty", split="whole")
    cols_to_be_embded = ['question_text']
    bs = 1024
    model_name = "meta-llama/Meta-Llama-3-8B"
    
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(
        dataloader, model_name, cols_to_be_embded
    )
    
    X = emb['question_text']
    y = dataset['z3']

    X = np.array(X) 
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_error = mean_squared_error(y_train, y_train_pred)
    print(f'Training Set Error (MSE): {train_error}')

    test_error = mean_squared_error(y_test, y_test_pred)
    print(f'Test Set Error (MSE): {test_error}')