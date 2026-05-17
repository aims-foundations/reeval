from huggingface_hub import snapshot_download

repo_id = "stair-lab/reeval"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="data",
)

print(f"Downloaded `{repo_id}` into `./{folder}`")
