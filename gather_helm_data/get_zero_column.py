import os
import re
import argparse
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
import pickle
from sentence_transformers import SentenceTransformer

scenario2pattern = {
    "mmlu": ["Question:", "Answer:"],
    "classic/babi_qa": ["Passage:", "Answer:"],
    "classic/bbq": ["Passage:", "Answer:"],
    "classic/blimp": None, # No fewshot examples
    "classic/bold": None, # No fewshot examples
    "classic/boolq": ["Passage:", "Answer:"],
    "classic/civil_comment": ["Passage:", "Answer:"],
    "classic/code": ["QUESTION:", "\n\n\nUse Standard Input format\n\nANSWER in Python code:\n"],
    "classic/commonsense": ["Question:", "Answer:"],
    "classic/copyright": None, # No fewshot examples
    "classic/disinfo": None, # Complex fewshot examples
    "classic/dyck_languague_np=3": ["Input:", None],
    "classic/entity_data_imputation": ["name:", "Answer:"],
    "classic/entity_matching": ["\n\n", "Answer:"],
    "classic/gsm:": ["Q:", "A:"],
    "classic/ice": None, # Not understand the fewshot structure
    "classic/imdb": ["Passage", "Sentiment"],
    "classic/legal_support": ["Passage:", "Answer:"],
    "classic/lsat_qa": ["Passage:", "Answer:"],
    "classic/math": ["Problem:", "Answer:"],
    "classic/mmlu": None, # skip MMLU in classic
    "classic/msmarco": ["Passage:", "Answer:"],
    "classic/narrative_qa": ["Passage:", "Answer:"],
    "classic/natural_qa": ["Question:", "Answer:"],
    "classic/quac": None, # Complex fewshot examples
    "classic/raft": None, # Complex fewshot examples
    "classic/real_toxicity_prompts": None, # No fewshot examples
    "classic/summarization_cnndm": ["Article:", None],
    "classic/summarization_xsum": ["Article:", None],
    "classic/synthetic_efficiency": None, # non-lantent construct
    "classic/synthetic_reasoning_natural": ["Rules:", None],
    "classic/synthetic_reasoning": [["Two results:", "Rules:"], "Target"],
    "classic/the_pile": None, # Not understand the fewshot structure
    "classic/truthful_qa": ["Question:", "Answer:"],
    "classic/twitter_aae": None, # No fewshot examples
    "classic/wikifact": ["\n\n", None],
}

def extract_last_question(pattern, text):
    if pattern is None:
        return text
    else:
        start_tag = pattern[0]
        end_tag = pattern[1]
        
        if end_tag is None:
            match = re.findall(f"({start_tag}.*)", text, re.DOTALL)
            if match:
                return match[-1].strip()
            else:
                return None
        
        elif isinstance(start_tag, list):
            start_tag1 = pattern[0]
            start_tag2 = pattern[1]
            match1 = re.findall(f"({start_tag1}.*?{end_tag})", text, re.DOTALL)
            match2 = re.findall(f"({start_tag2}.*?{end_tag})", text, re.DOTALL)
            if match1:
                return match1[-1].strip()
            elif match2:
                return match2[-1].strip()
            return None
        
        else:
            match = re.findall(f"({start_tag}.*?{end_tag})", text, re.DOTALL)
            if match:
                return match[-1].strip()
            else:
                return None

context = {
    "mmlu": """
        The following is a multiple choice (A, B, C, or D) question about %s from the Massive Multitask Language Understanding (MMLU) benchmark, 
        designed to measure ability in knowledge-intensive question answering across 57 domains.""",
    "classic/civil_comment": """
        Below is a true/false question based on a corresponding passage from the Civil Comments benchmark designed to measure the ability to identify 
        toxic comments. For this benchmark, a toxic comment is defined as one that attacks an individual's identity, includes insults or threats, 
        or contains explicit sexual content, obscene language, or other forms of severe toxicity. The question is about %s demographics group.
        """
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    data_folder = snapshot_download(repo_id="stair-lab/reeval_csv", repo_type="dataset")
    benchmark = args.dataset.split("/")[0]
    scenario = args.dataset.split("/")[-1]
    response = pickle.load(open(f"{data_folder}/{benchmark}/responses.pkl", "rb"))
    sid2scenario = pd.read_csv(f"{data_folder}/{benchmark}/scenarios.csv")
    question_key = pd.read_csv(f"{data_folder}/{benchmark}/instances.csv")
    model_key = pd.read_csv(f"{data_folder}/model_df.csv")
    flop = pd.read_csv(f"{data_folder}/FLOP.csv")
    flop["model_names_reeval"] = flop["model_names_reeval"].str.replace("_", "/")
    # merge model_key with flop: `model_names_reeval` should match `name` in model_key
    # not every name in model_key is in flop, and not every name in flop is in model_key
    # the resulting dataframe should has the same length as model_key
    model_key = pd.merge(model_key, flop, left_on="name", right_on="model_names_reeval", how="left")
    
    sid = sid2scenario[sid2scenario['name'] == scenario]['scenarios_id'].iloc[0]
    response = response[response["scenarios_id"] == sid]
    question_key = question_key[question_key["scenarios_id"] == sid]

    # remove the few-shot example from prompt
    pattern = scenario2pattern[args.dataset]
    n_questions = len(question_key["prompt"])
    print(f"dataset: {args.dataset}, number of question: {n_questions}")
    extract_example = extract_last_question(pattern, question_key["prompt"][0])
    print(f"prompt: {question_key['prompt'][0]}\nquestion_content: {extract_example}")
    breakpoint()
    last_questions = [extract_last_question(pattern, question_key["prompt"][i]) for i in range(n_questions)]
    question_key["question_content"] = last_questions

    # question_contexts = []
    # for i in range(n_questions):
    #     if args.dataset == "mmlu":
    #         subject_i = question_key["subject"][i]
    #         subject_i = subject_i.replace("_", " ")
    #         mmlu_context = context["mmlu"]
    #         question_contexts.append(mmlu_context % subject_i)
    #     elif args.dataset == "classic":
    #         if scenario == "civil_comment":
    #             breakpoint()
    #             demographics_group_i = question_key["demographics"][i]
    #             demographics_group_i = demographics_group_i.replace("_", " ")
    #             civil_comment_context = context["classic/civil_comment"]
    #             question_contexts.append(civil_comment_context % demographics_group_i)


    # question_key["question_contexts"] = question_contexts

    # merge data and instances by instance_id
    # instance_id in data is float, while in instances is int
    # but they are the same
    response["instance_id"] = response["instance_id"].astype(int)
    data = pd.merge(response, question_key, on="instance_id")

    # merge data with model_key by model_id
    data["model_id"] = data["model_id"].astype(int)
    data = pd.merge(data, model_key, on="model_id")   

    # save the data to parquet form
    save_dir = f"../data/another_repo/{scenario}"
    os.makedirs(save_dir, exist_ok=True)
    data.to_parquet(f"{save_dir}/data.parquet")

    # # upload the data to huggingface hub
    # api = HfApi()
    # # api.create_repo("stair-lab/reeval_another_repo", repo_type="dataset", exist_ok=True)
    # api.upload_file(
    #     repo_id="stair-lab/reeval_another_repo",
    #     path_in_repo=f"{scenario}/data.parquet",
    #     path_or_fileobj="data.parquet",
    #     repo_type="dataset"
    # )

    # # add a column for embedding (each embedding is a list of 4096 floats)
    # model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct" # "intfloat/e5-mistral-7b-instruct"
    # # model = LLM(model=model_name, enforce_eager=True)
    # model = SentenceTransformer(model_name, trust_remote_code=True)

    # pool = model.start_multi_process_pool()

    # embed_context_and_question_together = True
    # if embed_context_and_question_together:
    #     content_to_embed = [context + question for context, question in zip(question_contexts, last_questions)]
    #     embeddings = model.encode_multi_process(content_to_embed, pool, show_progress_bar=True, batch_size=32)
    #     model.stop_multi_process_pool(pool)

    #     embeddings = [emb.outputs.embedding for emb in embeddings]
    #     question_key[f"embeddings_{model_name.replace('/', '_')}"] = embeddings_last_questions
    #     question_key.to_parquet("question.parquet")

    #     api.upload_file(
    #         repo_id="stair-lab/reeval_another_repo",
    #         path_in_repo="mmlu/question.parquet",
    #         path_or_fileobj="question.parquet",
    #         repo_type="dataset"
    #     )

    # else:
    #     embeddings_question_contexts = model.encode_multi_process(question_contexts, pool, show_progress_bar=True, batch_size=32)
    #     embeddings_last_questions = model.encode_multi_process(last_questions, pool, show_progress_bar=True, batch_size=32)
    #     model.stop_multi_process_pool(pool)

    #     embeddings_question_contexts = [emb.tolist() for emb in embeddings_question_contexts]
    #     embeddings_last_questions = [emb.tolist() for emb in embeddings_last_questions]
    #     question_key[f"embeddings_question_contexts_{model_name.replace('/', '_')}"] = embeddings_question_contexts
    #     question_key[f"embeddings_last_questions_{model_name.replace('/', '_')}"] = embeddings_last_questions
    #     question_key.to_parquet("question_separated_emb.parquet")

    #     api.upload_file(
    #         repo_id="stair-lab/reeval_another_repo",
    #         path_in_repo="mmlu/question_separated_emb.parquet",
    #         path_or_fileobj="question_separated_emb.parquet",
    #         repo_type="dataset"
    #     )

    # # load the data from huggingface
    # data_folder = snapshot_download(repo_id="stair-lab/reeval_another_repo", repo_type="dataset")
    # data = pd.read_parquet(f"{data_folder}/mmlu/data.parquet", engine="fastparquet")
    # question_key = pd.read_parquet(f"{data_folder}/mmlu/question.parquet", engine="fastparquet")
    # question_key = pd.read_parquet(f"{data_folder}/mmlu/question_separated_emb.parquet", engine="fastparquet")