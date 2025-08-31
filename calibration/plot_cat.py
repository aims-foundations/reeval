import torch
from torchmetrics import AUROC
import matplotlib.pyplot as plt
import numpy as np

def auc_each_step(uhats, V,Y, idtor):
    aucs = []

    for i in range(uhats.shape[1]):
        auroc = AUROC(task="binary")
        probs = torch.sigmoid(uhats[:,i,:] @ V.T)
        aucs.append(auroc(probs[idtor],Y[idtor]))
    return aucs
        

def load_random_U_hat(dataset_name, test_taker_id):
    return torch.load(f"results/cat/random_thata_hats_{dataset_name}_{test_taker_id}.pt")

def load_adap_U_hat(dataset_name, test_taker_id):
    return torch.load(f"results/cat/adaptive_thata_hats_{dataset_name}_{test_taker_id}.pt")

dataset_name = "official_data"
random_thata_hats = []
adap_thata_hats = [] # num_test_taker * step * fac
for i in range(86):
    random_thata_hats.append(torch.stack(load_random_U_hat(dataset_name, i)))
    adap_thata_hats.append(torch.stack(load_adap_U_hat(dataset_name, i)))
random_thata_hats = torch.stack(random_thata_hats)
adap_thata_hats = torch.stack(adap_thata_hats)
data_idtor_test = torch.load("data/cat/data_idtor_test.pt")
V = torch.load("data/cat/V_true.pt")
test_data = torch.load("data/cat/test_data.pt")
data_idtor_test = torch.load("data/cat/data_idtor_test.pt")

aucs_random = auc_each_step(random_thata_hats, V, test_data, data_idtor_test)
aucs_adap = auc_each_step(adap_thata_hats, V, test_data, data_idtor_test)
plt.figure(figsize=(6, 5))
# plt.plot(np.arange(num_steps+1), (np.array(random_thata_hats) - theta_true) ** 2, label="random")
# plt.plot(np.arange(num_steps+1), (np.array(adaptive_thata_hats) - theta_true) ** 2, label="adaptive")

plt.plot( aucs_random, label="random")
plt.plot( aucs_adap, label="adaptive")
breakpoint()
plt.ylabel("auc")
plt.ylim(0.5, 0.7)
plt.title("auc adaptive testing")
plt.legend()
plt.show()
plt.savefig(f"plot/auc_adap_testing_{dataset_name}_{adap_thata_hats.shape[1] - 1}.png", dpi=600)
