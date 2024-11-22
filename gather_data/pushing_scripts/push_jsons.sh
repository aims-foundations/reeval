# List of dataset names
datasets=(
    "jsons/mmlu_json"
)

# Loop through each dataset and run the Python command
for dataset in "${datasets[@]}"; do
    echo "Uploading $dataset"
    huggingface-cli upload --repo-type dataset stair-lab/reeval_jsons $dataset $dataset &
done