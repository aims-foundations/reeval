import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from embed_text_package.embed_text import Embedder
from peft import AutoPeftModelForCausalLM
from ppo_reward_model import extract_score
from utils import MLP, plot_hist
from vllm import LLM, SamplingParams

