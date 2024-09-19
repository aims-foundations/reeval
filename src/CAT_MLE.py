import argparse
import torch
import random
import wandb
from tqdm import tqdm
from utils import item_response_fn_1PL, set_seed, save_state, load_state
import torch.optim as optim
from testtaker import SimulatedTestTaker
    
def CAT_owen(z3, unasked_question_list, theta_mean):
    z3_unasked = z3[unasked_question_list]
    z3_unasked = torch.tensor(z3_unasked)
    diff = abs(z3_unasked - theta_mean)
    return unasked_question_list[torch.argmin(diff)]

def CAT_fisher(z3, unasked_question_list, theta_mean):
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        theta = torch.tensor(theta_mean.item(), requires_grad=True)
        z_single = z3[unasked_question_index].clone().detach()    
        prob = item_response_fn_1PL(z_single, theta)
        hessian = prob * (1 - prob)
        fisher_info_list.append(hessian)
    index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[index_with_max_fisher_info]

def main(serial, strategy):
    set_seed(42)
    
    state = load_state("../data/synthetic/CAT_MLE/pre_cat.pt")
    z3 = state['z3']
    true_thetas = state['true_thetas']
    true_theta = true_thetas[serial]
    subset_question_num = state['subset_question_num']
    
    state_path = f"../data/synthetic/CAT_MLE/sweep/mle_{serial}_{strategy}.pt"
    question_num = z3.shape[0]
    init_question_index = random.randint(0, question_num-1)
    
    testtaker = SimulatedTestTaker(true_theta, model="1PL")
    
    theta_hat = torch.normal(
        mean=0.0, std=1.0, size=(1,), requires_grad=True, device='cuda'
    )
    optimizer = optim.Adam([theta_hat], lr=0.01)
    
    asked_question_list = [init_question_index]
    unasked_question_list = [
        i for i in range(question_num) if i != init_question_index
    ]
    asked_answer_list = [testtaker.ask(z3, init_question_index).cuda()]
    theta_hats = []
    
    pbar = tqdm(range(subset_question_num), desc=f"true theta: {true_theta}; epoch")
    for i in pbar:
        log_prob = 0
        for j, asked_question_index in enumerate(asked_question_list):
            prob = item_response_fn_1PL(z3[asked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob = log_prob + bernoulli.log_prob(asked_answer_list[j])
        
        loss = -log_prob / len(asked_question_list)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({"theta_hat": theta_hat.item()})
        theta_hats.append(theta_hat.item())

        if i == subset_question_num-1:
            break
            
        if strategy == "random": # random
            new_question_index = random.choice(unasked_question_list)
        elif strategy == "fisher": # fisher
            new_question_index = CAT_fisher(z3, unasked_question_list, theta_hat)
        # elif strategy=="owen":
        #     new_question_index = CAT_owen(z3, unasked_question_list, mean_theta)

        asked_question_list.append(new_question_index)
        unasked_question_list.remove(new_question_index)
        asked_answer_list.append(testtaker.ask(z3, new_question_index).cuda())
    
    save_state(
        state_path, 
        asked_question_list=asked_question_list,
        theta_hats=theta_hats, 
    )
    
if __name__ == "__main__":
    wandb.init(mode = "offline")
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=int)
    parser.add_argument("--strategy", type=str)
    args = parser.parse_args()
    
    main(args.serial, args.strategy)