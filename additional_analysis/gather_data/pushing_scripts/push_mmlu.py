import os

from huggingface_hub import HfApi

upload_api = HfApi()
for file in os.listdir(
    "/dfs/scratch1/nqduc/reeval/data/gather_data/crawl_real/jsons/mmlu_json"
):
    mmlu_file = (
        f"/dfs/scratch1/nqduc/reeval/data/gather_data/crawl_real/jsons/mmlu_json/{file}"
    )
    upload_api.upload_file(
        repo_id="stair-lab/reeval_jsons",
        repo_type="dataset",
        path_in_repo=f"jsons/mmlu_json/{file}",
        path_or_fileobj=mmlu_file,
        # run_as_future=True,
    )
