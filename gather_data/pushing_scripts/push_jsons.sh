# List of dataset names
datasets=(
    "jsons/babi_qa_json"
    "jsons/bbq_json"
    "jsons/blimp_json"
    "jsons/bold_json"
    "jsons/boolq_json"
    "jsons/civil_comments_json"
    "jsons/code_json"
    "jsons/commonsense_json"
    "jsons/copyright_json"
    "jsons/disinfo_json"
    "jsons/dyck_language_np=3_json"
    "jsons/entity_data_imputation_json"
    "jsons/entity_matching_json"
    "jsons/gsm_json"
    "jsons/ice_json"
    "jsons/imdb_json"
    "jsons/legal_support_json"
    "jsons/lsat_qa_json"
    "jsons/math_json"
    "jsons/mmlu_json"
    "jsons/msmarco_json"
    "jsons/narrative_qa_json"
    "jsons/natural_qa_json"
    "jsons/quac_json"
    "jsons/raft_json"
    "jsons/real_toxicity_prompts_json"
    "jsons/summarization_cnndm_json"
    "jsons/summarization_xsum_json"
    "jsons/synthetic_efficiency_json"
    "jsons/synthetic_reasoning_json"
    "jsons/synthetic_reasoning_natural_json"
    "jsons/the_pile_json"
    "jsons/truthful_qa_json"
    "jsons/twitter_aae_json"
    "jsons/wikifact_json"
)

# Loop through each dataset and run the Python command
for dataset in "${datasets[@]}"; do
    echo "Uploading $dataset"
    huggingface-cli upload --repo-type dataset stair-lab/reeval_jsons $dataset $dataset &
done