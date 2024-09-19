import os
import numpy as np
import pandas as pd
import torch
from utils import item_response_fn_3PL, item_response_fn_2PL, item_response_fn_1PL, item_response_fn_1PL_cheat

class SimulatedTestTaker():
    def __init__(self, theta=None, model="1PL"):
        self.model = model
        if theta==None:
            self.ability = torch.normal(mean=0.0, std=1.0, size=(1,))
        else:
            self.ability = theta

    def ask(self, Z, question_index):
        if self.model == "3PL":
            prob = item_response_fn_3PL(
                *Z[question_index, :], self.ability
            )
        elif self.model == "2PL":
            prob = item_response_fn_2PL(
                *Z[question_index, :], self.ability
            )
        elif self.model == "1PL":
            prob = item_response_fn_1PL(
                Z[question_index], self.ability
            )
        return torch.distributions.Bernoulli(prob).sample()
    
    def get_ability(self):
        return self.ability

class CheatingTestTaker():
    def __init__(self, true_theta, cheat_gain=0, model="1PL"):
        self.model = model
        self.true_ability = true_theta
        self.cheat_ability = true_theta + cheat_gain

    def ask(self, Z, contamination, question_index):
        if self.model == "1PL":
            prob = item_response_fn_1PL_cheat(
                Z[question_index], contamination[question_index], self.true_ability, self.cheat_ability
            )
        return torch.distributions.Bernoulli(prob).sample()
    
    def get_ability(self):
        return self.true_ability, self.cheat_ability

class RealTestTaker():
    def __init__(self, question_text_list, model_string):
        data_df = pd.read_csv(f"../data/real/pre_irt_data/eval/eval_{model_string}_result.csv")
        prompt_score_dict = dict(zip(data_df['prompt'], data_df['score']))
        self.score_list = [prompt_score_dict.get(text, None) for text in question_text_list]
        assert None not in self.score_list

    def ask(self, Z, question_index):
        assert len(Z) == len(self.score_list)
        return torch.tensor(self.score_list[question_index])
    
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    question_num = 500
    testtaker_num = 1000

    # 1PL
    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    
    response_matrix = np.zeros((testtaker_num, question_num))
    true_theta = np.zeros(testtaker_num)
    for i in range(testtaker_num):
        new_testtaker = SimulatedTestTaker(model="1PL")
        true_theta[i] = new_testtaker.get_ability().item()
        
        for j in range(question_num):
            response_matrix[i, j] = new_testtaker.ask(z3, j).item()
            
    save_dir = "../data/synthetic/response_matrix/"
    os.makedirs(save_dir, exist_ok=True)

    response_df = pd.DataFrame(response_matrix.astype(int))
    response_df.insert(0, '', [f'testtaker_{i}' for i in range(testtaker_num)])
    response_df.to_csv(os.path.join(save_dir, "synthetic_matrix_1PL.csv"), index=False)

    true_theta_df = pd.DataFrame(true_theta, columns=["true_theta"])
    true_theta_df.to_csv(os.path.join(save_dir, "true_theta.csv"), index=False)
    
    Z_df = pd.DataFrame(z3.numpy(), columns=["z3"])
    Z_df.to_csv(os.path.join(save_dir, "true_Z_1PL.csv"), index=False)
    
    # 3PL
    z1 = torch.distributions.Beta(0.5, 4).sample((question_num,))
    z2 = torch.distributions.LogNormal(0, 1).sample((question_num,))
    Z = torch.vstack((z1, z2, z3)).T

    response_matrix = np.zeros((testtaker_num, question_num))
    true_theta = np.zeros(testtaker_num)
    for i in range(testtaker_num):
        new_testtaker = SimulatedTestTaker(model="3PL")
        true_theta[i] = new_testtaker.get_ability().item()
        
        for j in range(question_num):
            response_matrix[i, j] = new_testtaker.ask(Z, j).item()
            
    save_dir = "../data/synthetic/response_matrix/"
    os.makedirs(save_dir, exist_ok=True)

    response_df = pd.DataFrame(response_matrix.astype(int))
    response_df.insert(0, '', [f'testtaker_{i}' for i in range(testtaker_num)])
    response_df.to_csv(os.path.join(save_dir, "synthetic_matrix_3PL.csv"), index=False)
    
    Z_df = pd.DataFrame({
        "z1": z1.numpy(),
        "z2": z2.numpy(),
        "z3": z3.numpy()
    })
    Z_df.to_csv(os.path.join(save_dir, "true_Z_3PL.csv"), index=False)
