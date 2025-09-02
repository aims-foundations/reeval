from util import get_all_model_meta_info
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
all_meta = get_all_model_meta_info()

# --- prep ---
df = all_meta.copy()
df["#Params (B)"] = pd.to_numeric(df["#Params (B)"], errors="coerce")
df = df.dropna(subset=["#Params (B)", "Official Providers"])

# make provider labels readable (handles True/False/0/1/strings)
df["Official Providers"] = df["Official Providers"].astype(str)

def plot_hist_by_group(
    data: pd.DataFrame,
    x_col: str,
    group_col: str,
    bins: int = 30,
    alpha: float = 0.5,
    figsize=(10, 6),
    outfile: str = None,
    title: str = None
):
    plt.figure(figsize=figsize)

    # consistent bins across groups (important for % comparability)
    x_values = data[x_col].to_numpy()
    bin_edges = np.histogram_bin_edges(x_values, bins=bins)

    for label, g in data.groupby(group_col):
        x = g[x_col].to_numpy()
        # weights so each group's bars sum to 100% (percent per bin)
        weights = np.ones_like(x, dtype=float) * (100.0 / len(x))
        plt.hist(x, bins=bin_edges, weights=weights, alpha=alpha, label=str(label))

    plt.xlabel(x_col)
    plt.ylabel("Percentage (%)")
    if title:
        plt.title(title)
    plt.legend(title=group_col)
    plt.grid(True, linestyle="--", alpha=0.6)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# 1) Overlapping transparent histograms by each provider label (as-is)
plot_hist_by_group(
    df,
    x_col="#Params (B)",
    group_col="Official Providers",
    bins=30,
    alpha=0.5,
    outfile="hist_params_by_provider_percent.png",
    title="Model Params by Provider (percentage, overlapping)"
)


# Ensure datetime and filter rows we can use
df = all_meta.copy()
df["Upload To Hub Date"] = pd.to_datetime(df["Upload To Hub Date"], errors="coerce")
df = df.dropna(subset=["Upload To Hub Date", "Official Providers"]).copy()

# Normalize provider labels strictly to "True"/"False"
df["Official Providers"] = df["Official Providers"].astype(str).map(
    {"True": "True", "False": "False"}
).fillna("False")

# Monthly bucket
df["Month"] = df["Upload To Hub Date"].dt.to_period("M").dt.to_timestamp()

# Count uploads per (provider, month)
counts = df.groupby(["Official Providers", "Month"]).size().unstack("Month", fill_value=0)

# Fill missing months in the overall range so both providers share same timeline
all_months = pd.date_range(df["Month"].min(), df["Month"].max(), freq="MS")
counts = counts.reindex(columns=all_months, fill_value=0)

# Convert to percentages over time PER PROVIDER (rows sum to 100%)
percentages = counts.div(counts.sum(axis=1), axis=0) * 100

# For side-by-side bars per month, we want months as rows, providers as columns
plot_df = percentages.T.sort_index()   # shape: [months x providers]

# Plot
ax = plot_df.plot(
    kind="bar",
    figsize=(14, 6),
    alpha=0.8
)
ax.set_ylabel("Percentage of that provider’s total uploads (%)")
ax.set_xlabel("Month")
ax.set_title("Monthly share of uploads (each provider sums to 100% across time)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Official Providers")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/plot/hist_date_by_provider_percent.png", dpi=300, bbox_inches="tight")

plt.show()


def plot_categorical_by_provider_percent(
    df: pd.DataFrame,
    cat_col: str,
    provider_col: str = "Official Providers",
    top_k: int = 15,
    min_count: int | None = None,   # keep any category with total >= min_count
    min_pct: float | None = None,   # keep any category with total% >= min_pct (0-100)
    title: str = None,
    figsize=(14, 6),
    outfile: str | None = None,
    alpha: float = 0.85,
):
    data = df.copy()
    data = data.dropna(subset=[cat_col, provider_col])

    # Normalize provider labels to "True"/"False"
    data[provider_col] = data[provider_col].astype(str).map(
        {"True": "True", "False": "False"}
    ).fillna("False")

    # Raw counts per (provider, category)
    counts = data.groupby([provider_col, cat_col]).size().unstack(fill_value=0)

    # Total counts across providers (used for selection and sorting)
    total_by_cat = counts.sum(axis=0).sort_values(ascending=False)
    total_n = int(total_by_cat.sum())

    # Build keep list
    keep = set(total_by_cat.index[:top_k]) if top_k is not None else set(total_by_cat.index)

    if min_count is not None:
        keep |= set(total_by_cat[total_by_cat >= min_count].index)

    if min_pct is not None:
        keep |= set(total_by_cat[(total_by_cat / total_n * 100) >= min_pct].index)

    # Ensure deterministic order: by total freq (desc), with "Other" last if used
    keep_sorted = [c for c in total_by_cat.index if c in keep]

    # Collapse the rest into "Other"
    dropped = [c for c in total_by_cat.index if c not in keep]
    counts_reduced = counts.copy()
    if dropped:
        counts_reduced["Other"] = counts_reduced[dropped].sum(axis=1)
        counts_reduced = counts_reduced.drop(columns=dropped)
        cat_order = keep_sorted + ["Other"]
    else:
        cat_order = keep_sorted

    # Row-normalize to percentages (each provider sums to 100)
    percentages = counts_reduced.div(counts_reduced.sum(axis=1), axis=0) * 100

    # Reindex columns to our chosen order and pivot for plotting (categories x providers)
    plot_df = percentages[cat_order].T

    # Plot
    ax = plot_df.plot(kind="bar", figsize=figsize, alpha=alpha)
    ax.set_ylabel("Percentage within provider (%)")
    ax.set_xlabel(cat_col)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=provider_col)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Optionally return the data used (helpful for debugging or tables)
    return {
        "counts": counts,
        "counts_reduced": counts_reduced,
        "percentages": percentages,
        "plot_df": plot_df
    }

# Architecture distribution (per-provider %)
plot_categorical_by_provider_percent(
    all_meta,
    cat_col="Architecture",
    top_k=20,
    title="Architecture distribution by Official Provider (percentage) showing top 20",
    outfile="results/plot/arch_by_provider_pct.png"
)

# Base Model distribution (per-provider %)
plot_categorical_by_provider_percent(
    all_meta,
    cat_col="Base Model",
    top_k=20,
    title="Base Model distribution by Official Provider (percentage) showing top 20",
    outfile="results/plot/base_model_by_provider_pct.png"
)



# selected_arch = [
#     "LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM", "Gemma2ForCausalLM",
#     "MixtralForCausalLM", "Qwen2Model", "Phi3ForCausalLM", "?",
#     "GemmaForCausalLM", "PhiForCausalLM", "GPTNeoXForCausalLM", "GPT2LMHeadModel",
#     "Qwen2MoeForCausalLM", "GraniteForCausalLM", "T5ForConditionalGeneration",
#     "MllamaForConditionalGeneration"
# ]

# # Filter down
# df = all_meta.copy()
# df = df[df["Architecture"].isin(selected_arch)].dropna(subset=["Official Providers"])

# # Normalize provider labels strictly
# df["Official Providers"] = df["Official Providers"].astype(str).map(
#     {"True": "True", "False": "False"}
# ).fillna("False")

# # Count (provider, architecture)
# counts = df.groupby(["Official Providers", "Architecture"]).size().unstack(fill_value=0)

# # Convert to % per provider (rows sum to 100)
# percentages = counts.div(counts.sum(axis=1), axis=0) * 100
# plot_df = percentages.T.loc[selected_arch]  # keep order

# # Plot
# ax = plot_df.plot(kind="bar", figsize=(14, 6), alpha=0.85)
# ax.set_ylabel("Percentage within provider (%)")
# ax.set_xlabel("Architecture")
# ax.set_title("Architecture Distribution by Official Provider (restricted to selected names)")
# plt.xticks(rotation=45, ha="right")
# plt.legend(title="Official Providers")
# plt.grid(axis="y", linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig("results/plot/arch_by_provider_pct.png", dpi=300, bbox_inches="tight")
# plt.show()