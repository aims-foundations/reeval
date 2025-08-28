import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import joblib
import joblib as jbl
import numpy as np
import pandas as pd
import requests
from datasets import Dataset
from datasets import load_dataset

from zarth_utils.logger import logging_info
from zarth_utils.general_utils import makedir_if_not_exist

dir_data_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(dir_data_base, exist_ok=True)
dir_data_raw = os.path.join(dir_data_base, "raw")
os.makedirs(dir_data_raw, exist_ok=True)

helm_datasets = [
    "commonsense", "gsm", "legalbench",
    "math", "med_qa", "mmlu",
    # "narrative_qa", "natural_qa", "wmt_14", # not binary accuracy
]
glue_datasets = [
    # "cola", # not binary accuracy
    "mrpc", "rte", "sst2", "mnli", "qnli",
    # "qqp", # this dataset always throws an error when loading
]
openllm_datasets = [
    "ifeval",
    "openllm_math",
    "mmlu_pro", "arc_challenge",
    "bbh", "gpqa", "musr"
]
all_datasets = helm_datasets + glue_datasets + openllm_datasets + ["imagenet"]


def get_scores(dataset_name):
    dir_scores = os.path.join(dir_data_base, "scores")
    os.makedirs(dir_scores, exist_ok=True)
    path_scores = os.path.join(dir_scores, "%s.jbl" % dataset_name)

    if os.path.exists(path_scores):
        scores = jbl.load(path_scores)
    else:
        if dataset_name in helm_datasets:
            helm = HelmLite(tasks=[dataset_name])
            helm.download_and_check()
            dataset = helm.get_datasets()
            scores = np.array(dataset["acc"]).T
            jbl.dump(helm.models, os.path.join(dir_scores, "%s_models.jbl" % dataset_name))

        elif dataset_name in openllm_datasets:
            openllm = OpenLLM(tasks=[dataset_name])
            openllm.download_and_check()
            dataset = openllm.get_datasets()
            scores = np.array(dataset["acc"]).T
            jbl.dump(openllm.models, os.path.join(dir_scores, "%s_models.jbl" % dataset_name))

        elif dataset_name in glue_datasets:
            all_preds, gold_labels = load_glue_predictions(dataset_name)
            scores = np.zeros([all_preds.shape[0], all_preds.shape[1]])
            for m in range(all_preds.shape[0]):
                scores[m] = (all_preds[m].argmax(-1) == gold_labels)

        else:
            raise NotImplementedError

        jbl.dump(scores, path_scores)

    if dataset_name == "imagenet":
        real_acc = scores.mean(-1)
        scores = scores[real_acc > 0.3]

    scores = scores.astype(np.float32)

    assert len(np.unique(scores)) == 2
    # if len(np.unique(scores)) != 2:
    #     print(f"Warning: scores of {dataset_name} are not binary, convert to binary.")
    #     scores = (scores > np.mean(scores)).astype(np.float32)

    real_acc = scores.mean(-1)
    scores = scores[real_acc > 1e-8]

    return scores


def load_glue_predictions(task="cola", model_family=None):
    all_tasks = ["cola", "mnli", "mrpc", "qnli", "rte", "sst2"]
    assert task in all_tasks
    all_model_family = ["BERT", "GPT", "INSTRUCT_GPT", "OPENAI"]
    if model_family is None:
        model_family = all_model_family
    elif isinstance(model_family, str):
        model_family = [model_family]
    else:
        assert isinstance(model_family, list)
    if task == "mnli":
        gold_label = np.array(load_dataset("SetFit/mnli", split="validation")["label"])
    else:
        gold_label = np.array(load_dataset("glue", task, split="validation")["label"])

    all_pred = []
    for f in model_family:
        assert f in all_model_family
        pred_f = np.load(
            os.path.join(dir_data_raw, "model_preds_by_family_glue", f, "%s_preds.npy" % task),
            allow_pickle=True
        )
        all_pred.append(pred_f)
    all_pred = np.concatenate(all_pred, axis=0)
    assert all_pred.shape[1] == len(gold_label), (all_pred.shape, len(gold_label))
    return all_pred, gold_label


class Benchmark(ABC):
    def __init__(self):
        self.dir_data = None
        self.models = None
        self.tasks = None

    def download_and_check(self):
        all_task_texts = dict()
        all_task_labels = dict()
        all_task_model_perf = defaultdict(dict)
        for task in self.tasks:
            for i, model in enumerate(self.models):
                try:
                    meta_info = self.get_meta_infos(model, task)
                except Exception as e:
                    self.models = [m for m in self.models if m != model]
                    logging_info(str(e))
                    logging_info("Model %s not found for task %s" % (model, task))
                    logging_info("Remove model %s" % model)
                    continue
                all_task_texts[task] = meta_info["text"]
                all_task_labels[task] = meta_info["label"]
                all_task_model_perf[task][model] = meta_info["acc"]

        for task_name in self.tasks:
            task_texts = all_task_texts[task_name]
            for model_name in self.models:
                for idx in task_texts.index:
                    if idx not in all_task_model_perf[task_name][model_name].index:
                        self.models = [model for model in self.models if model != model_name]
                        logging_info("Remove model %s" % model_name)
                        break

    @abstractmethod
    def get_meta_infos(self, model_name, task_name, *args, **kwargs) -> pd.DataFrame:
        pass

    def get_texts(self, task_name) -> pd.Series:
        return self.get_meta_infos(self.models[0], task_name)["text"]

    def get_labels(self, task_name) -> pd.Series:
        return self.get_meta_infos(self.models[0], task_name)["label"]

    def get_accuracies(self, model_name, task_name) -> pd.Series:
        return self.get_meta_infos(model_name, task_name)["acc"]

    def get_datasets(self) -> Dataset:
        all_task_texts = dict()
        all_task_labels = dict()
        all_task_model_perf = defaultdict(dict)
        for task in self.tasks:
            for i, model in enumerate(self.models):
                try:
                    meta_info = self.get_meta_infos(model, task)
                except FileNotFoundError:
                    logging_info("Model %s not found for task %s" % (model, task))
                    continue
                all_task_texts[task] = meta_info["text"]
                all_task_labels[task] = meta_info["label"]
                all_task_model_perf[task][model] = meta_info["acc"]

        ret_texts, ret_perfs, ret_labels, ret_tasks = [], [], [], []
        for task_name in self.tasks:
            task_texts = all_task_texts[task_name]
            task_labels = all_task_labels[task_name]

            ret_texts += list(task_texts.values)
            ret_labels += list(task_labels.values)
            ret_tasks += [task_name] * len(task_texts)
            ret_perfs += [
                [
                    all_task_model_perf[task_name][model_name].loc[idx]
                    for model_name in self.models
                ]
                for idx in task_texts.index
            ]

        ret_dataset = Dataset.from_dict(
            {
                "text": ret_texts,
                "acc": ret_perfs,
                "label": ret_labels,
                "task": ret_tasks,
            }
        )

        return ret_dataset

    def get_leaderboard_all(self):
        leaderboard = {task: np.zeros(len(self.models)) for task in self.tasks}
        task_size = defaultdict(int)
        for task_name in self.tasks:
            for i, model_name in enumerate(self.models):
                acc = self.get_accuracies(model_name, task_name).values
                leaderboard[task_name][i] = acc.mean()
                task_size[task_name] = len(acc)

        leaderboard = pd.DataFrame(leaderboard, index=self.models)
        leaderboard["mean-task"] = leaderboard.mean(axis=1)
        mean_sample = []
        for model in self.models:
            tot, n = 0, 0
            for task in self.tasks:
                tot += leaderboard.loc[model][task] * task_size[task]
                n += task_size[task]
            mean_sample.append(tot / n)
        leaderboard["mean-sample"] = mean_sample

        return leaderboard


def download_json(url, filename):
    response = requests.get(url)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = response.json()

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        return True
    else:
        return False


class HelmLite(Benchmark):
    def __init__(self, models=None, tasks=None):
        super().__init__()
        self.dir_data = os.path.join(dir_data_raw, "helm_lite")
        if models is None:
            self.models = [
                "01-ai_yi-34b",
                "01-ai_yi-6b",
                "AlephAlpha_luminous-base",
                "AlephAlpha_luminous-extended",
                "AlephAlpha_luminous-supreme",
                "ai21_j2-grande",
                "ai21_j2-jumbo",
                "anthropic_claude-2.0",
                "anthropic_claude-2.1",
                "anthropic_claude-instant-1.2",
                "anthropic_claude-v1.3",
                "cohere_command",
                "cohere_command-light",
                "google_text-bison@001",
                "google_text-unicorn@001",
                "meta_llama-2-13b",
                "meta_llama-2-70b",
                "meta_llama-2-7b",
                "meta_llama-65b",
                "mistralai_mistral-7b-v0.1",
                "mistralai_mixtral-8x7b-32kseqlen",
                "openai_gpt-3.5-turbo-0613",
                "openai_gpt-4-0613",
                "openai_gpt-4-1106-preview",
                "openai_text-davinci-002",
                "openai_text-davinci-003",
                "tiiuae_falcon-40b",
                "tiiuae_falcon-7b",
                "writer_palmyra-x-v2",
                "writer_palmyra-x-v3",
                "microsoft_phi-2",
                "mistralai_mistral-medium-2312",
                "allenai_olmo-7b",
                "google_gemma-7b",
                "meta_llama-3-70b",
                "meta_llama-3-8b",
                "mistralai_mixtral-8x22b",
                "qwen_qwen1.5-14b",
                "qwen_qwen1.5-32b",
                "qwen_qwen1.5-72b",
                "qwen_qwen1.5-7b",
                "anthropic_claude-3-haiku-20240307",
                "anthropic_claude-3-opus-20240229",
                "anthropic_claude-3-sonnet-20240229",
                "databricks_dbrx-instruct",
                "deepseek-ai_deepseek-llm-67b-chat",
                "mistralai_mistral-large-2402",
                "mistralai_mistral-small-2402",
                "cohere_command-r",
                "cohere_command-r-plus",
                "mistralai_mistral-7b-instruct-v0.3",
                "openai_gpt-4-turbo-2024-04-09",
                "openai_gpt-4o-2024-05-13",
                "qwen_qwen1.5-110b-chat",
                "qwen_qwen2-72b-instruct",
                "snowflake_snowflake-arctic-instruct",
                "01-ai_yi-large-preview",
                "anthropic_claude-3-5-sonnet-20240620",
                "google_gemini-1.0-pro-002",
                "google_gemini-1.5-flash-001",
                "google_gemini-1.5-pro-001",
                "ai21_jamba-instruct",
                "google_gemma-2-27b-it",
                "google_gemma-2-9b-it",
                "microsoft_phi-3-medium-4k-instruct",
                "mistralai_mistral-large-2407",
                "mistralai_open-mistral-nemo-2407",
                "openai_gpt-4o-mini-2024-07-18",
                "meta_llama-3.1-405b-instruct-turbo",
                "meta_llama-3.1-70b-instruct-turbo",
                "meta_llama-3.1-8b-instruct-turbo",
                "ai21_jamba-1.5-large",
                "ai21_jamba-1.5-mini",
                "meta_llama-3.2-11b-vision-instruct-turbo",
                "meta_llama-3.2-90b-vision-instruct-turbo",
                "qwen_qwen2.5-72b-instruct-turbo",
                "qwen_qwen2.5-7b-instruct-turbo",
                "anthropic_claude-3-5-sonnet-20241022",
                "google_gemini-1.5-flash-002",
                "google_gemini-1.5-pro-002",
                "openai_gpt-4o-2024-08-06",
                "meta_llama-3.3-70b-instruct-turbo",
                "upstage_solar-pro-241126",
            ]
        else:
            self.models = models
        if tasks is None:
            self.tasks = [
                "commonsense",
                "gsm",
                "legalbench",
                "math",
                "med_qa",
                "mmlu",
                "narrative_qa",
                "natural_qa",
                "wmt_14",
            ]
        else:
            self.tasks = tasks
        self.dir_predictions = os.path.join(self.dir_data, "predictions")

    def download(self, model_name, task_name, json_name):
        task2metric = {
            "narrative_qa": "f1_score",
            "natural_qa": "f1_score",
            "commonsense": "exact_match",
            "mmlu": "exact_match",
            "math": "math_equiv_chain_of_thought",
            "gsm": "final_number_exact_match",
            "legalbench": "quasi_exact_match",
            "med_qa": "quasi_exact_match",
            "wmt_14": "bleu_4",
        }
        if ":" in task_name and task_name.split(":")[0] in task2metric.keys():
            subtasks = [task_name]
            task_name = task_name.split(":")[0]
        else:
            assert task_name in task2metric.keys()
            subtasks = {
                "commonsense": [
                    "commonsense:dataset=openbookqa,method=multiple_choice_joint,"
                ],
                "gsm": [
                    "gsm:",
                ],
                "legalbench": [
                    "legalbench:subset=abercrombie,",
                    "legalbench:subset=corporate_lobbying,",
                    "legalbench:subset=function_of_decision_section,",
                    "legalbench:subset=international_citizenship_questions,",
                    "legalbench:subset=proa,",
                ],
                "math": [
                    "math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,",
                ],
                "med_qa": ["med_qa:"],
                "mmlu": [
                    "mmlu:subject=abstract_algebra,method=multiple_choice_joint,",
                    "mmlu:subject=college_chemistry,method=multiple_choice_joint,",
                    "mmlu:subject=computer_security,method=multiple_choice_joint,",
                    "mmlu:subject=econometrics,method=multiple_choice_joint,",
                    "mmlu:subject=us_foreign_policy,method=multiple_choice_joint,",
                ],
                "narrative_qa": ["narrative_qa:"],
                "natural_qa": [
                    "natural_qa:mode=closedbook,",
                    "natural_qa:mode=openbook_longans,",
                ],
                "wmt_14": [
                    "wmt_14:language_pair=cs-en,",
                    "wmt_14:language_pair=de-en,",
                    "wmt_14:language_pair=fr-en,",
                    "wmt_14:language_pair=hi-en,",
                    "wmt_14:language_pair=ru-en,",
                ],
            }[task_name]

        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            if not os.path.exists(path_save):
                for version in range(20):
                    url = (
                            "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.%d.0/%s/%s"
                            % (version, file_name, json_name)
                    )
                    if download_json(url, path_save):
                        break
            if task_name == "gsm" and not os.path.exists(path_save):
                file_name = subtask + "model=%s" % model_name
                path_save = os.path.join(self.dir_predictions, file_name, json_name)
                if not os.path.exists(path_save):
                    for version in range(20):
                        url = (
                                "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.%d.0/%s/%s"
                                % (
                                    version,
                                    subtask + "model=%s,stop=none" % model_name,
                                    json_name,
                                )
                        )
                        if download_json(url, path_save):
                            break

        n = 0
        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            res = json.load(open(path_save, "r"))
            n += len(res)

        ret = []
        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            res = json.load(open(path_save, "r"))
            for i in range(len(res)):
                if "instance_id" in res[i].keys():
                    res[i]["instance_id"] = "%s-%s" % (subtask, res[i]["instance_id"])
                if "id" in res[i].keys():
                    res[i]["id"] = "%s-%s" % (subtask, res[i]["id"])
                metric_name = task2metric[task_name]
                if "stats" in res[i].keys() and metric_name in res[i]["stats"]:
                    res[i]["stats"]["acc"] = (
                            1.0
                            * res[i]["stats"][metric_name]
                        # without following lines, acc will be different from the original aggregated score
                        # * n
                        # / len(res)
                        # / len(subtasks)
                    )
            ret += res
        return ret

    def get_meta_infos(self, model_name, task_name) -> pd.DataFrame:
        dir_save = os.path.join(self.dir_data, "perfs")
        makedir_if_not_exist(dir_save)
        path_save = os.path.join(dir_save, "%s_%s.jbl" % (model_name, task_name))

        if os.path.exists(path_save):
            ret = jbl.load(path_save)
        else:
            instances = self.download(model_name, task_name, "instances.json")
            predictions = self.download(
                model_name, task_name, "display_predictions.json"
            )
            ret = {
                "id": [],
                "text": [],
                "label": [],
                "acc": [],
            }
            for ins, pred in zip(instances, predictions):
                assert ins["id"] == pred["instance_id"]
                idx = "%s-%s" % (task_name, ins["id"])
                ret["id"].append(idx)
                # todo: label is not loaded
                ret["label"].append(-1)
                ret["acc"].append(pred["stats"]["acc"])
                ref_text = " ".join(
                    [
                        "%s. %s" % (chr(i + 65), ref["output"]["text"])
                        for i, ref in enumerate(ins["references"])
                    ]
                )
                ret["text"].append(ins["input"]["text"] + "\n" + ref_text)
            ret = pd.DataFrame(ret)
            ret.set_index("id", inplace=True)
            joblib.dump(ret, path_save)

        return ret


class OpenLLM(Benchmark):
    def __init__(self, models=None, tasks=None):
        super(OpenLLM, self).__init__()
        self.dir_data = os.path.join(dir_data_raw, "openllm")

        if models is None:
            official_providers_dataset = load_dataset("open-llm-leaderboard/official-providers")
            official_providers = official_providers_dataset["train"][0]["CURATED_SET"]
            leaderboard_dataset = load_dataset("open-llm-leaderboard/contents")
            all_models = leaderboard_dataset["train"]["fullname"]
            self.models = [m for m in all_models if m.split("/")[0] in official_providers]
        else:
            self.models = models
        if tasks is None:
            self.tasks = [
                "ifeval",
                "openllm_math",
                "mmlu_pro",
                "arc_challenge",
                "bbh",
                "gpqa",
                "musr"
            ]
        else:
            self.tasks = tasks
        self.dir_predictions = os.path.join(self.dir_data, "predictions")

    def get_meta_infos(self, model_name, task_name) -> pd.DataFrame:
        task2metric = {
            "ifeval": "prompt_level_strict_acc",
            "openllm_math": "exact_match",
            "mmlu_pro": "acc",
            "gpqa": "acc_norm",
            "musr": "acc_norm",
            "bbh": "acc_norm",
            "arc_challenge": "acc_norm",
        }
        task2subtasks = {
            "ifeval": ["ifeval"],
            "openllm_math": [
                'math_geometry_hard',
                'math_intermediate_algebra_hard',
                'math_num_theory_hard',
                'math_prealgebra_hard',
                'math_precalculus_hard'
            ],
            "mmlu_pro": ["mmlu_pro"],
            "arc_challenge": ["arc_challenge"],
            "bbh": [
                "bbh_boolean_expressions",
                "bbh_causal_judgement",
                "bbh_date_understanding",
                "bbh_disambiguation_qa",
                "bbh_formal_fallacies",
                "bbh_geometric_shapes",
                "bbh_hyperbaton",
                "bbh_logical_deduction_five_objects",
                "bbh_logical_deduction_seven_objects",
                "bbh_logical_deduction_three_objects",
                "bbh_movie_recommendation",
                "bbh_navigate",
                "bbh_object_counting",
                "bbh_penguins_in_a_table",
                "bbh_reasoning_about_colored_objects",
                "bbh_ruin_names",
                "bbh_salient_translation_error_detection",
                "bbh_snarks",
                "bbh_sports_understanding",
                "bbh_temporal_sequences",
                "bbh_tracking_shuffled_objects_five_objects",
                "bbh_tracking_shuffled_objects_seven_objects",
                "bbh_tracking_shuffled_objects_three_objects",
                "bbh_web_of_lies"],
            "gpqa": [
                "gpqa_diamond",
                "gpqa_extended",
                "gpqa_main"],
            "musr": [
                "musr_murder_mysteries",
                "musr_object_placements",
                "musr_team_allocation", ]
        }
        model_name = model_name.replace("/", "__")
        ret = {
            "id": [],
            "text": [],
            "label": [],
            "acc": [],
        }
        for subtask in task2subtasks[task_name]:
            data = load_dataset(
                "open-llm-leaderboard/%s-details" % model_name,
                name="%s__leaderboard_%s" % (model_name, subtask),
                split="latest",
            )
            ret["id"] += ["%s-%d" % (subtask, i) for i in data["doc_id"]]
            ret["label"] += data["target"]
            ret["acc"] += data[task2metric[task_name]]
            # todo: text is not loaded
            ret["text"] += ["" for _ in data["doc"]]

        ret = pd.DataFrame(ret)
        ret.set_index("id", inplace=True)

        return ret



# Initialize with default models and tasks
openllm = OpenLLM()

# openllm = OpenLLM(models=custom_models, tasks=custom_tasks)

# Download and check data availability
openllm.download_and_check()

# Get the dataset
dataset = openllm.get_datasets()
print(f"Dataset contains {len(dataset)} samples")
print(f"Available models: {openllm.models}")