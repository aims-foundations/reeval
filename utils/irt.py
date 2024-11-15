import torch
from torch import nn
from tqdm import tqdm
from torch import optim


class IRT(nn.Module):
    def __init__(self, n_questions, n_testtaker, D=1, PL=1):
        super(IRT, self).__init__()
        self.D = D
        self.PL = PL
        self.ability = nn.Parameter(torch.randn(n_testtaker, D), requires_grad=True)
        self.difficulty = nn.Parameter(torch.randn(n_questions), requires_grad=True)
        
        if D == 1:
            self.register_buffer('loading_factor', torch.ones(n_questions, D))
        elif D > 1:
            self.loading_factor = torch.randn(n_questions, D)
            self.loading_factor = nn.Parameter(self.loading_factor, requires_grad=True)
        else:
            raise ValueError(f'D={D} is not supported')
        
        if PL == 1:
            self.register_buffer('disciminatory', torch.ones(n_questions))
            self.register_buffer('guessing', torch.zeros(n_questions))
        elif PL == 2:
            self.disciminatory = nn.Parameter(torch.exp(torch.randn(n_questions)), requires_grad=True)
            self.register_buffer('guessing', torch.zeros(n_questions))
        elif PL == 3:
            self.disciminatory = nn.Parameter(torch.exp(torch.randn(n_questions)), requires_grad=True)
            self.guessing = nn.Parameter(torch.randn(n_questions), requires_grad=True)
        else:
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
    
    def fit(self, method="mle", max_epoch=3000, response_matrix=None):
        if method == "mle":
            self.mle(max_epoch, response_matrix)
        else:
            raise ValueError(f'{method} is not supported')
        
    def forward(self):
        ability = self.get_abilities()
        difficulty = self.get_difficulty()
        disciminatory = self.get_disciminatory()
        guessing = self.get_guessing()
        loading_factor = self.get_loading_factor()
        
        return self.compute_prob(ability, difficulty, disciminatory, guessing, loading_factor)

    def mle(self, max_epoch, response_matrix):
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        pbar = tqdm(range(max_epoch))
        for _ in pbar:
            prob_matrix = self.forward()

            mask = response_matrix != -1
            masked_response_matrix = response_matrix.flatten()[mask.flatten()]
            masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

            berns = torch.distributions.Bernoulli(masked_prob_matrix)
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
        mean_difficulty = torch.mean(self.difficulty)
        std_difficulty = torch.std(self.difficulty)
        difficulty = (self.difficulty - mean_difficulty) / std_difficulty
        return difficulty
            
    def get_disciminatory(self):
        return torch.relu(self.disciminatory)
    
    def get_guessing(self):
        if self.PL == 3:
            return torch.sigmoid(self.guessing)
        return self.guessing
    
    def get_loading_factor(self):
        return torch.softmax(self.loading_factor, dim=1)

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
