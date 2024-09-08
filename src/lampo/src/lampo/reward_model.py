"""
This is reward model server main process.
"""

import importlib
import random
from abc import ABC, abstractmethod

from . import utils


def get_reward_model(model_name, config):
    """
    This function is used for initialize reward model.
    """
    if model_name == "random":
        rm = RandomRewardModel(config=config)
    elif model_name != "":
        model_class = utils.import_submodule(model_name)

        if not issubclass(model_class, RewardModelTemplate):
            raise NotImplementedError(
                "Please inherit your reward model"
                "from llmppo.reward_model.RewardModelTemplate"
            )
        rm = model_class(config=config)
    else:
        raise NotImplementedError("Please pass the reward model to --model")

    return rm


class RewardModelTemplate(ABC):
    """
    Reward Model Template
    """

    @abstractmethod
    def __init__(self, config):
        """
        Initialize necessary things
        """

    @abstractmethod
    async def compute(self, messages):
        """
        This is the main function for implement reward computation.
        """

    @abstractmethod
    def load(self):
        """
        Load model (if necessary)
        """

    @abstractmethod
    def unload(self):
        """
        Unload model
        """


class RandomRewardModel(RewardModelTemplate):
    """
    Implementation of Random Reward Model
    """

    def __init__(self, config):
        """
        No need to initialize
        """
        super(RandomRewardModel).__init__()

    async def compute(self, messages):
        """
        This is implementation of random reward model.
        """
        num_rewards = len(messages)
        return [random.uniform(0, 1) for _ in range(num_rewards)]

    def load(self):
        """
        No need to load model
        """
        return

    def unload(self):
        """
        No need to unload model
        """
        return
