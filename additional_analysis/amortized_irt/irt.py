import copy

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn, optim
from tqdm import tqdm

from .network import MLP


class IRT(nn.Module):
    def __init__(
        self,
        D=1,
        PL=1,
        device="cpu",
        report_to=None,
    ):
        super().__init__()
        self.D = D
        self.PL = PL
        self.device = device
        self.report_to = report_to
        self.tol = 1e-6

    @classmethod
    def compute_prob(
        cls,
        ability,
        difficulty,
        disciminatory=None,
        guessing=None,
        loading_factor=None,
    ):
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
        if ability.shape == difficulty.shape:
            return guessing + (1 - guessing) * torch.sigmoid(
                disciminatory * (ability * loading_factor) + difficulty
            )
        else:
            return guessing + (1 - guessing) * torch.sigmoid(
                disciminatory * (ability * loading_factor).sum(-1) + difficulty
            )

    def init_parameters(
        self,
        response_matrix,
        amortize_item=False,
        amortize_student=False,
        amortized_question_hyperparams=None,
        amortized_model_hyperparams=None,
    ):
        self.n_students, self.n_questions = response_matrix.shape[:2]
        self.amortize_item = amortize_item
        self.amortize_student = amortize_student
        self.amortized_question_hyperparams = amortized_question_hyperparams
        self.amortized_model_hyperparams = amortized_model_hyperparams

        if self.amortize_student:
            assert self.amortized_model_hyperparams is not None

            self.ability_nn = MLP(
                **self.amortized_model_hyperparams,
                output_dim=self.D,
                device=self.device,
            )
            self.ability_mask = None
        else:
            self.ability = nn.Parameter(
                torch.randn(self.n_students, self.D, device=self.device),
                requires_grad=True,
            )
            self.ability_mask = None

        if self.D == 1:
            self.register_buffer(
                "loading_factor",
                torch.ones(self.n_questions, self.D, device=self.device),
            )
        if self.PL == 1:
            self.register_buffer(
                "disciminatory",
                torch.ones(self.n_questions, device=self.device),
            )
        if self.PL == 1 or self.PL == 2:
            self.register_buffer(
                "guessing",
                torch.zeros(self.n_questions, device=self.device),
            )

        if self.amortize_item:
            assert self.amortized_question_hyperparams is not None
            self.item_parameters_nn = MLP(
                **self.amortized_question_hyperparams,
                output_dim=3 + self.D,
                device=self.device,
            )

        else:
            self.difficulty = nn.Parameter(
                torch.randn(self.n_questions, device=self.device),
                requires_grad=True,
            )

            if self.D > 1:
                self.loading_factor = torch.randn(
                    self.n_questions, self.D, device=self.device
                )
                self.loading_factor = nn.Parameter(
                    self.loading_factor, requires_grad=True
                )
            elif self.D < 1:
                raise ValueError(f"D={self.D} is not supported")

            if self.PL == 2:
                self.disciminatory = nn.Parameter(
                    torch.exp(
                        torch.randn(self.n_questions, device=self.device),
                    ),
                    requires_grad=True,
                )
            elif self.PL == 3:
                self.disciminatory = nn.Parameter(
                    torch.exp(
                        torch.randn(self.n_questions, device=self.device),
                    ),
                    requires_grad=True,
                )
                self.guessing = nn.Parameter(
                    torch.randn(self.n_questions, device=self.device),
                    requires_grad=True,
                )
            elif self.PL != 1:
                raise ValueError(f"PL={self.PL} is not supported")

    def fit(
        self,
        method="mle",
        max_epoch=3000,
        response_matrix=None,
        embedding=None,
        model_features=None,
        amortized_question_hyperparams=None,
        amortized_model_hyperparams=None,
    ):
        if embedding is not None:
            amortize_item = True
            assert amortized_question_hyperparams is not None, (
                "Please provide hyperparameters for item amortization"
            )
        else:
            amortize_item = False

        if model_features is not None:
            amortize_student = True
            assert amortized_model_hyperparams is not None, (
                "Please provide hyperparameters for student amortization"
            )
        else:
            amortize_student = False

        with torch.no_grad():
            self.init_parameters(
                response_matrix=response_matrix,
                amortize_item=amortize_item,
                amortize_student=amortize_student,
                amortized_question_hyperparams=amortized_question_hyperparams,
                amortized_model_hyperparams=amortized_model_hyperparams,
            )

        if method == "mle":
            return self.mle(
                max_epoch=max_epoch,
                response_matrix=response_matrix,
                embedding=embedding,
                model_features=model_features,
            )
        elif method == "em":
            return self.em(
                max_epoch=max_epoch,
                response_matrix=response_matrix,
                embedding=embedding,
                model_features=model_features,
            )
        else:
            raise ValueError(f"{method} is not supported")

    def forward(self):
        ability = self.get_abilities()
        difficulty = self.get_difficulties()
        disciminatory = self.get_disciminatory()
        guessing = self.get_guessing()
        loading_factor = self.get_loading_factor()

        return self.compute_prob(
            ability, difficulty, disciminatory, guessing, loading_factor
        )

    def em(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        embedding=None,
        model_features=None,
        n_mc_samples: int = 100,
    ):
        self.em_item(
            max_epoch=max_epoch,
            response_matrix=response_matrix,
            embedding=embedding,
            n_mc_samples=n_mc_samples,
        )
        self.em_ability(
            max_epoch=max_epoch,
            response_matrix=response_matrix,
            embedding=embedding,
            model_features=model_features,
        )

    def em_item(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        embedding=None,
        n_mc_samples: int = 64,
        batch_size=4,
    ):
        theta_nodes, gh_weights = np.polynomial.hermite_e.hermegauss(n_mc_samples)
        theta_nodes = np.sqrt(2) * theta_nodes  # Transform nodes
        theta_nodes = torch.tensor(theta_nodes, device=response_matrix.device)
        # theta_nodes, weights = np.polynomial.hermite_e.hermegauss(n_mc_samples)
        # theta_nodes = torch.tensor(theta_nodes, device=response_matrix.device)

        theta_matrices = theta_nodes[:, None, None]
        theta_matrices = theta_matrices.repeat(1, self.n_students, self.D)
        # >>> n_mc_samples x n_students x D

        weights = torch.tensor(gh_weights, device=response_matrix.device)
        weights = weights / np.sqrt(np.pi)  # Adjust weights for the normal density
        weights = weights / torch.sum(weights)  # Optionally normalize if needed
        # weights = torch.tensor(weights, device=response_matrix.device)
        # weights = weights / torch.sum(weights)
        # >>> n_mc_samples x 1

        if self.amortize_item:
            optimizer = optim.Adam(
                self.item_parameters_nn.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
            )
        else:
            parameters = [self.difficulty]
            if self.PL > 1:
                parameters.append(self.disciminatory)
            if self.PL > 2:
                parameters.append(self.guessing)
            if self.D > 1:
                parameters.append(self.loading_factor)
            optimizer = optim.Adam(parameters, lr=0.005)

        mask = response_matrix != -1

        pbar = tqdm(range(max_epoch))
        previous_parameters = None
        previous_loss = None

        for iteration in pbar:
            if self.amortize_item:
                self.item_parameters = self.item_parameters_nn(embedding)

            if iteration > 0:
                previous_parameters = copy.deepcopy(parameters)
                previous_loss = copy.deepcopy(loss.item())

            difficulty = self.get_difficulties()
            disciminatory = self.get_disciminatory()
            guessing = self.get_guessing()
            loading_factor = self.get_loading_factor()

            # instead forwarding on the full theta_matrices,
            # do it in batches of row and accumulate the gradients
            optimizer.zero_grad()
            loss = []
            for batch_start in range(0, self.n_students, batch_size):
                batch_end = min(batch_start + batch_size, self.n_students)

                prob_matrices = self.compute_prob(
                    theta_matrices[:, batch_start:batch_end],
                    difficulty,
                    disciminatory,
                    guessing,
                    loading_factor,
                )

                local_mask = mask[batch_start:batch_end]

                masked_prob_matrix = prob_matrices  # [:, local_mask]
                # >>> n_mc_samples x batch_size x n_questions

                obs = response_matrix[batch_start:batch_end]  # [local_mask]
                obs = obs[None, :, :].repeat(n_mc_samples, 1, 1)
                # >>> n_mc_samples x batch_size x n_questions

                berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
                ll = berns.log_prob(obs)

                ll_sum = ll.sum(dim=2)
                # >>> n_mc_samples x batch_size

                weights_local = weights[:, None].repeat(1, obs.shape[1])
                # >>> n_mc_samples x batch_size

                log_marginal_prob = torch.logsumexp(
                    ll_sum + torch.log(weights_local), dim=0
                )
                loss.append(-log_marginal_prob)

            loss = torch.concatenate(loss).mean()
            loss.backward()
            optimizer.step()

            if self.report_to is not None:
                wandb.log({"loss_item": loss.item()})

            if iteration > 0:
                params_norm_diff = 0
                for p, pp in zip(parameters, previous_parameters):
                    params_norm_diff += torch.norm(p - pp, p=2).item()

                loss_diff = torch.abs(loss - previous_loss).item()

                # compute gradient norm
                grad_norm = 0
                for p in parameters:
                    grad_norm += torch.norm(p.grad, p=2).item()
                pbar.set_postfix(
                    {
                        "grad_norm": grad_norm,
                        "params_norm_diff": params_norm_diff,
                        "loss_diff": loss_diff,
                    }
                )

                if (
                    params_norm_diff < self.tol
                    and loss_diff < self.tol
                    and grad_norm < self.tol
                ):
                    break

        item_parms = self.get_item_parameters().cpu().detach().tolist()
        item_parms = np.array(item_parms)
        difficulties = item_parms[:, 0]
        difficulties = pd.DataFrame(difficulties)
        difficulties.to_csv("difficulties_python.csv", index=False, header=False)

    def em_ability(
        self,
        max_epoch: int,
        response_matrix: torch.Tensor,
        embedding=None,
        model_features=None,
    ):
        if self.amortize_student:
            optimizer = optim.Adam(
                self.ability_nn.parameters(), lr=1e-3, weight_decay=1e-4
            )
        else:
            optimizer = optim.Adam([self.ability], lr=0.001)

        pbar = tqdm(range(max_epoch))
        if self.amortize_item:
            with torch.no_grad():
                self.item_parameters = self.item_parameters_nn(embedding)

        if self.amortize_student:
            self.ability_mask = model_features[:, 0] != -1
            response_matrix = response_matrix[self.ability_mask]

        mask = response_matrix != -1
        masked_response_matrix = response_matrix[mask]

        previous_parameters = None
        previous_loss = None

        for iteration in pbar:
            if self.amortize_student:
                self.ability = self.ability_nn(model_features)

            if iteration > 0:
                previous_parameters = copy.deepcopy(self.ability)
                previous_loss = copy.deepcopy(loss.item())

            optimizer.zero_grad()

            prob_matrix = self.forward()

            if self.amortize_student:
                masked_prob_matrix = prob_matrix[self.ability_mask][mask]
                abilities = self.ability[self.ability_mask]
            else:
                masked_prob_matrix = prob_matrix[mask]
                abilities = self.ability

            berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
            loss = -berns.log_prob(masked_response_matrix).mean()

            # encourage the ability to have mean 0 and std 1
            # mean_ability = torch.mean(abilities, dim=0)
            # std_ability = torch.std(abilities, dim=0)
            # loss = (
            #     loss
            #     + torch.abs(mean_ability).mean()
            #     + torch.abs(std_ability - 1).mean()
            # )

            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss_ability": loss.item()})
            if self.report_to is not None:
                wandb.log({"loss_ability": loss.item()})

            if iteration > 0:
                params_norm_diff = torch.norm(
                    self.ability - previous_parameters, p=2
                ).item()
                loss_diff = torch.abs(loss - previous_loss).item()

                # compute gradient norm
                grad_norm = torch.norm(self.ability.grad, p=2).item()
                pbar.set_postfix(
                    {
                        "grad_norm": grad_norm,
                        "params_norm_diff": params_norm_diff,
                        "loss_diff": loss_diff,
                    }
                )

                if (
                    params_norm_diff < self.tol
                    and loss_diff < self.tol
                    and grad_norm < self.tol
                ):
                    break

    def get_abilities(self):
        if self.ability_mask is None and self.amortize_student:
            raise ValueError("Please fit the model first")

        if self.ability_mask is not None:
            ability = self.ability[self.ability_mask]
        else:
            ability = self.ability

        # mean_ability = torch.mean(ability, dim=0)
        # std_ability = torch.std(ability, dim=0)
        # ability = (self.ability - mean_ability) / std_ability

        return ability

    def get_difficulties(self):
        if self.amortize_item:
            difficulty = self.item_parameters[..., 0]
        else:
            difficulty = self.difficulty

        return difficulty
        # mean_difficulty = torch.mean(difficulty)
        # std_difficulty = torch.std(difficulty)
        # return (difficulty - mean_difficulty) / std_difficulty

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
                self.get_difficulties(),
                self.get_disciminatory(),
                self.get_guessing(),
            ],
            dim=-1,
        )
        return torch.cat([item_params, self.get_loading_factor()], dim=-1)

    @classmethod
    def apply_item_constrains(cls, item_parameters, D, PL):
        difficulty = item_parameters[..., 0:1]
        disciminatory = item_parameters[..., 1:2]
        guessing = item_parameters[..., 2:3]
        loading_factor = item_parameters[..., 3:]

        mean_difficulty = torch.mean(difficulty)
        std_difficulty = torch.std(difficulty)
        difficulty = (difficulty - mean_difficulty) / std_difficulty

        if PL == 1:
            disciminatory = torch.ones_like(disciminatory)
        else:
            disciminatory = torch.relu(disciminatory)

        if PL < 3:
            guessing = torch.zeros_like(guessing)
        else:
            guessing = torch.sigmoid(guessing)

        if D == 1:
            loading_factor = torch.ones_like(loading_factor)
        else:
            loading_factor = torch.softmax(loading_factor, dim=-1)

        return torch.cat(
            [difficulty, disciminatory, guessing, loading_factor],
            dim=-1,
        )

    @classmethod
    def apply_student_constrains(cls, abilities, model_features):
        ability_mask = model_features[:, 0] != -1

        mean_ability = torch.mean(abilities[ability_mask], dim=0)
        std_ability = torch.std(abilities[ability_mask], dim=0)

        # replace -1 with torch nan
        abilities[~ability_mask] = torch.nan
        abilities = (abilities - mean_ability) / std_ability
        return abilities
