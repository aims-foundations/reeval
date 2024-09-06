import torch
import random
from fit_theta import fit_theta_mle
from utils import item_response_fn_1PL, set_seed, save_state, load_state
from argparse import ArgumentParser

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

def main(question_num, subset_question_num, testtaker_num):
    print(f'question_num: {question_num}')
    print(f'subset_question_num: {subset_question_num}')
    print(f'testtaker_num: {testtaker_num}')
    
    state_path = f"../data/synthetic/CAT/mle_{question_num}_{subset_question_num}_{testtaker_num}.pt"
    # state = load_state(state_path)
    # if state:
    #     z3 = state['z3']
    #     true_thetas = state['true_thetas']
    #     asked_question_lists = state['asked_question_lists']
    #     unasked_question_lists = state['unasked_question_lists']
    #     asked_answer_lists = state['asked_answer_lists']
    #     theta_means = state['theta_means']
    #     start_epoch = state['epoch'] + 1

    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    init_question_index = random.randint(0, question_num - 1)
    asked_question_lists = [[init_question_index] for _ in range(testtaker_num)]
    unasked_question_lists = [[i for i in range(question_num) if i != init_question_index] for _ in range(testtaker_num)]
    
    true_thetas = torch.normal(mean=0.0, std=1.0, size=(testtaker_num,))
    probs = item_response_fn_1PL(z3[init_question_index], true_thetas)
    responses = torch.distributions.Bernoulli(probs).sample()
    
    asked_answer_lists = [[r] for r in responses]
    
    theta_means = [[] for _ in range(testtaker_num)]
    for epoch in range(subset_question_num-1):
        print(f'\nepoch: {epoch+1}')
        
        new_question_indexs = []
        for i in range(testtaker_num):
            print(f'testtaker: {i+1}')
            asked_question_tensor = torch.tensor(asked_question_lists[i])
            asked_answer_tensor = torch.tensor(asked_answer_lists[i])
            z3_tensor = z3.clone().detach()

            mean_theta = fit_theta_mle(
                z3_tensor, 
                asked_question_tensor, 
                asked_answer_tensor
            )
            theta_means[i].append(mean_theta)

            if i%2 == 0: # random
                new_question_index = random.choice(unasked_question_lists[i])
            elif i%2 == 1: # fisher
                new_question_index = CAT_fisher(z3, unasked_question_lists[i], mean_theta)
            # elif strategy=="owen":
            #     new_question_index = CAT_owen(z3, unasked_question_list, mean_theta)
            new_question_indexs.append(new_question_index)
            asked_question_lists[i].append(new_question_index)
            unasked_question_lists[i].remove(new_question_index)
        
        assert len(new_question_indexs) == len(true_thetas)
        probs = item_response_fn_1PL(z3[new_question_indexs], true_thetas)
        responses = torch.distributions.Bernoulli(probs).sample()
        for j in range(len(asked_answer_lists)):
            asked_answer_lists[j].append(responses[j])
    
        save_state(
            state_path, 
            z3 = z3,
            true_thetas=true_thetas,
            asked_question_lists=asked_question_lists,
            theta_means=theta_means, 
        )
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--subset_question_num", type=int, default=100)
    parser.add_argument("--testtaker_num", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    
    main(args.question_num, args.subset_question_num, args.testtaker_num)
