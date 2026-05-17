import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

bundles.icml2024()  # Data from the table

datasets = [
    "boolq",
    "syn_reason",
    "mmlu",
    "wikifact",
    "math",
    "quac",
    "civil_comments",
    "babi_qa",
    "raft",
    "bbq",
    "lsat_qa",
    "commonsense",
    "truthful_qa",
    "syn_reason_nat",
    "entity_match",
    "bold",
    "dyck",
    "twitter_aae",
    "imdb",
    "narrative_qa",
    "legal_support",
    "ent_data_imp",
    "airbench",
    "combined_data",
]

ctt_auc = [
    0.51,
    0.50,
    0.50,
    0.50,
    0.50,
    0.52,
    0.51,
    0.52,
    0.50,
    0.51,
    0.52,
    0.49,
    0.51,
    0.49,
    0.52,
    0.51,
    0.51,
    0.50,
    0.50,
    0.50,
    0.50,
    0.49,
    0.50,
    0.50,
]
ctt_std = [
    0.07,
    0.07,
    0.06,
    0.07,
    0.06,
    0.07,
    0.07,
    0.07,
    0.07,
    0.07,
    0.06,
    0.07,
    0.08,
    0.05,
    0.08,
    0.06,
    0.07,
    0.06,
    0.07,
    0.07,
    0.06,
    0.06,
    0.07,
    0.06,
]

irt_auc = [
    0.81,
    0.74,
    0.87,
    0.87,
    0.83,
    0.82,
    0.63,
    0.83,
    0.79,
    0.71,
    0.69,
    0.53,
    0.71,
    0.73,
    0.67,
    0.75,
    0.78,
    0.98,
    0.82,
    0.91,
    0.61,
    0.94,
    0.85,
    0.82,
]
irt_std = [
    0.07,
    0.12,
    0.05,
    0.05,
    0.11,
    0.07,
    0.08,
    0.05,
    0.06,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.10,
    0.10,
    0.07,
    0.02,
    0.11,
    0.04,
    0.06,
    0.03,
    0.05,
    0.08,
]

# Create bar plot
x = np.arange(len(datasets))  # the label locations
width = 0.4  # width of the bars

with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(9 / 1.5, 3 / 1.5))
    ax.bar(
        x,
        irt_auc,
        width,
        yerr=irt_std,
        capsize=5,
        label="Rasch",
        alpha=0.5,
        color="blue",
    )
    ax.bar(
        x,
        ctt_auc,
        width,
        yerr=ctt_std,
        capsize=5,
        label="Mean score",
        alpha=0.5,
        color="red",
    )

    ax.set_ylabel("AUC-ROC")
    ax.set_xticks(x)
    ax.set_ylim([0.3, 1])
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    plt.savefig("generalization.png", dpi=300, bbox_inches="tight")
