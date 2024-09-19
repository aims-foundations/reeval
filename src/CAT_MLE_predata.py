from utils import set_seed, save_state
import torch

if __name__ == "__main__":
    seed = 42
    question_num = 10000
    subset_question_num = 1000
    testtaker_num = 500
    path = "../data/synthetic/CAT_MLE/pre_cat.pt"
    
    set_seed(seed)
    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    true_thetas = torch.normal(mean=0.0, std=1.0, size=(testtaker_num,))
    save_state(path, z3=z3, true_thetas=true_thetas, subset_question_num=subset_question_num)

    