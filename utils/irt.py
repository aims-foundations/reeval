import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from .network import MLP

class IRT(nn.Module):
    def __init__(
        self, n_questions, n_testtaker, D=1, PL=1, amortize_item=False, amortized_model_hyperparams=None
    ):
        super(IRT, self).__init__()
        self.D = D
        self.PL = PL
        self.amortize_item = amortize_item
        self.ability = nn.Parameter(torch.randn(n_testtaker, D), requires_grad=True)

        if D == 1:
            self.register_buffer('loading_factor', torch.ones(n_questions, D))
        if PL == 1:
            self.register_buffer('disciminatory', torch.ones(n_questions))
        if PL == 1 or PL == 2:
            self.register_buffer('guessing', torch.zeros(n_questions))
                
                
        if amortize_item:
            assert amortized_model_hyperparams is not None
            self.item_parameters_nn = MLP(
                **amortized_model_hyperparams,
                output_dim=3+D,
            )
            
        else:
            self.difficulty = nn.Parameter(torch.randn(n_questions), requires_grad=True)
        
            if D > 1:
                self.loading_factor = torch.randn(n_questions, D)
                self.loading_factor = nn.Parameter(self.loading_factor, requires_grad=True)
            elif D < 1:
                raise ValueError(f'D={D} is not supported')
            
            if PL == 2:
                self.disciminatory = nn.Parameter(torch.exp(torch.randn(n_questions)), requires_grad=True)
            elif PL == 3:
                self.disciminatory = nn.Parameter(torch.exp(torch.randn(n_questions)), requires_grad=True)
                self.guessing = nn.Parameter(torch.randn(n_questions), requires_grad=True)
            elif PL != 1:
                raise ValueError(f'PL={PL} is not supported')
    
    @classmethod
    def compute_prob(cls, ability, difficulty, disciminatory=None, guessing=None, loading_factor=None):
        ab_shape = list(ability.size())
        df_shape = list(difficulty.size())
        if disciminatory is None:
            dc_shape = df_shape[:-1] + [ab_shape[-2]] + [df_shape[-1]]
            disciminatory = torch.ones(dc_shape).to(ability)
        if guessing is None:
            gs_shape = df_shape[:-1] + [ab_shape[-2]] + [df_shape[-1]]
            guessing = torch.zeros(gs_shape).to(ability)
        if loading_factor is None:
            lf_shape = ab_shape[:-1] + [1] + [ab_shape[-1]]
            loading_factor = torch.ones(lf_shape).to(ability)
        
        ability = ability[..., None, :]
        difficulty = difficulty[..., None, :]
        return (
            guessing + (1 - guessing) * 
            torch.sigmoid(disciminatory * (ability * loading_factor).sum(-1) + difficulty)
        )
    
    def fit(self, method="mle", max_epoch=3000, response_matrix=None, embedding=None):
        if method == "mle":
            self.mle(max_epoch, response_matrix, embedding)
        elif method == "em":
            self.em(max_epoch, response_matrix, embedding)
        else:
            raise ValueError(f'{method} is not supported')
        
    def forward(self):
        ability = self.get_abilities()
        difficulty = self.get_difficulty()
        disciminatory = self.get_disciminatory()
        guessing = self.get_guessing()
        loading_factor = self.get_loading_factor()
        
        return self.compute_prob(ability, difficulty, disciminatory, guessing, loading_factor)

    def mle(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        embedding=None,
    ):  
        if self.amortize_item:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        else:
            optimizer = optim.Adam(self.parameters(), lr=0.01)

        pbar = tqdm(range(max_epoch))
        for _ in pbar:
            if self.amortize_item:
                self.item_parameters = self.item_parameters_nn(embedding)

            prob_matrix = self.forward()

            mask = response_matrix != -1
            masked_response_matrix = response_matrix.flatten()[mask.flatten()]
            masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

            berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
            loss = -berns.log_prob(masked_response_matrix).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})
    
    def em(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        num_node: int = 64,
    ):
        self.em_item(max_epoch, response_matrix, num_node)
        self.em_ability(max_epoch, response_matrix)

    def em_item(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        n_mc_samples: int = 64,
    ):
        n_testtaker, num_item = response_matrix.shape
        
        theta_nodes, weights = np.polynomial.hermite.hermgauss(n_mc_samples)
        theta_nodes = torch.tensor(theta_nodes, device=response_matrix.device)
        
        theta_matrices = theta_nodes[:, None, None].repeat(1, n_testtaker, self.D)
        # >>> n_mc_samples x n_testtaker x D
        
        weights = torch.tensor(weights, device=response_matrix.device)
        weights = weights / torch.sum(weights)
        # >>> n_mc_samples

        parameters = [self.difficulty]
        if self.PL > 1:
            parameters.append(self.disciminatory)
        if self.PL > 2:
            parameters.append(self.guessing)
        if self.D > 1:
            parameters.append(self.loading_factor)
        optimizer = optim.Adam(parameters, lr=0.01)

        pbar = tqdm(range(max_epoch))
        for _ in pbar:
            difficulty = self.get_difficulty()
            disciminatory = self.get_disciminatory()
            guessing = self.get_guessing()
            loading_factor = self.get_loading_factor()
            
            prob_matrices = self.compute_prob(
                theta_matrices,
                difficulty,
                disciminatory,
                guessing,
                loading_factor
            )

            mask = response_matrix != -1
            masked_response_matrix = response_matrix[mask]
            masked_prob_matrix = prob_matrices[:, mask]

            berns = torch.distributions.Bernoulli(
                probs=(masked_prob_matrix * weights[:,None]).sum(0)
            )
            loss = -berns.log_prob(masked_response_matrix).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": loss.item()})

    def em_ability(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
    ):
        optimizer = optim.Adam([self.ability], lr=0.01)
        pbar = tqdm(range(max_epoch))
        for _ in pbar:
            prob_matrix = self.forward()

            mask = response_matrix != -1
            masked_response_matrix = response_matrix.flatten()[mask.flatten()]
            masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

            berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
            loss = -berns.log_prob(masked_response_matrix).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})

    def get_abilities(self):
        mean_ability = torch.mean(self.ability, dim=0)
        std_ability = torch.std(self.ability, dim=0)
        ability = (self.ability - mean_ability) / std_ability
        return ability
    
    def get_difficulty(self):
        if self.amortize_item:
            difficulty = self.item_parameters[..., 0]
        else:
            difficulty = self.difficulty

        mean_difficulty = torch.mean(difficulty)
        std_difficulty = torch.std(difficulty)
        return (difficulty - mean_difficulty) / std_difficulty
            
    def get_disciminatory(self):
        if self.PL > 1:
            if self.amortize_item:
                disciminatory = self.item_parameters[..., 1]
            else:
                disciminatory = self.disciminatory
                
            return torch.relu(disciminatory)
        else:
            return self.disciminatory
            
    def get_guessing(self):
        if self.PL == 3:
            if self.amortize_item:
                guessing = self.item_parameters[..., 2]
            else:
                guessing = self.guessing
                
            return torch.sigmoid(guessing)
        else:
            return self.guessing
    
    def get_loading_factor(self):
        if self.D > 1:
            if self.amortize_item:
                loading_factor = self.item_parameters[..., 3:]
            else:
                loading_factor = self.loading_factor
                
            return torch.softmax(loading_factor, dim=1)
        else:
            return self.loading_factor

    def get_item_parameters(self):
        item_params = torch.stack(
            [
                self.get_difficulty(),
                self.get_disciminatory(),
                self.get_guessing()
            ], 
            dim=-1
        )
        return torch.cat([item_params, self.get_loading_factor()], dim=-1)