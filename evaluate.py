import os
import re
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from predict import get_label_space
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random, full]")
    parser.add_argument("--shots", type=int, default=-1, help="zero/few shot")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model_name", type=str, default="chat", help="[chat]")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xxl", help="")
    parser.add_argument('--slm_model_name', type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="1", help="the type of prompt")
    parser.add_argument("--retriver_type", type=str, default="random", help="the type of retriver: ['random], 'bm25', ...] ")
    parser.add_argument("--test_path", type=str, default="test_0_10.csv", help="the path of multimodal data")
    parser.add_argument("--demo_label_prefix_type", type=str, default="1", help="the prefix of label in the demonstration ")
    parser.add_argument("--label_map_type", type=str, default="0", help="whether has label_map ")
    parser.add_argument("--use_context", action="store_true", help="whether use context for ScienceQA")
    return parser.parse_args()



# Define a function to extract the label from a string
def extract_label(string):
    pattern = r'{\[(.*?)\]}'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return "NONE"


def extract_labels(task, dataset, df):
    ill_formed_idx, diff_idx = [], []
    
    if task == 'MASC' or task=="MSA":
        index = df['original_index']
        true_labels = df["label_text"]
        pred_labels = df["prediction"]
    elif task == 'MRE':
        if dataset == 'MNRE':
            true_labels = df["label_text"]
            pred_labels = df["prediction"]
    elif task == "MHMR":
        true_labels = df["label_text"]
        pred_labels = df["prediction"]
    elif task == "MSR":
        true_labels = df["label_text"]
        pred_labels = df["prediction"]
    elif task=="QA":
        if dataset == "ScienceQA" or dataset == "ScienceQA_no_image" or dataset=="ScienceQA_1" or dataset=="ScienceQA_no_context":
            
            true_labels = df["answer"]
            pred_labels = df["predictions_index"]    

    else:
        raise NotImplementedError

    if task != "absa":
        print("+++++++++++++++++++++++++++++++++++++++++++")
        for i in range(len(pred_labels)):
            pred = str(pred_labels[i]).lower().strip()
            if task == 'MASC' or task=="MSA":
                if dataset == 'TumEmo':
                    if pred not in ["angry", "bored", "calm", "fear", "happy", "love", "sad"]:
                        print('the index is {}, and pred is {}'.format(index[i], pred))
                elif dataset == 'MASAD':
                    if pred not in ['negative', 'positive']:
                        print('the index is {}, and pred is {}'.format(index[i], pred))
                        
                elif dataset =='MOSI_2' or dataset == "MOSEI_2":
                    if pred not in ['negative', 'positive']:
                        print('the index is {}, and pred is {}'.format(index[i], pred))
                elif dataset == 'MOSI_7' or dataset == "MOSEI_7":
                    if pred not in ["strongly positive", "positive", "weakly positive", "neutral", "weakly negative", "negative", "strongly negative"]:
                        print('the index is {}, and pred is {}'.format(index[i], pred))
                        
                else:
                    if pred not in ['negative', 'positive', 'neutral']:
                        print('the index is {}, and pred is {}'.format(index[i], pred))
                        
            
        true_labels = [str(i).lower().strip() for i in true_labels]
        pred_labels = [str(i).lower().strip() for i in pred_labels]
        pred_counter = Counter(pred_labels)
        gold_counter = Counter(true_labels)
        # print(classification_report(true_labels, pred_labels, zero_division=0))

        print("Gold:")
        print_counter(gold_counter)
        print("Pred:")
        print_counter(pred_counter)

    return true_labels, pred_labels, ill_formed_idx

def print_counter(freq_dict):
    total_len = sum(freq_dict.values())
    for item, freq in freq_dict.items():
        print(f"{item}: {freq} ({freq/total_len*100:.2f}%)")


def process_tuple_f1(labels, predictions, verbose=False):
    tp, fp, fn = 0, 0, 0
    epsilon = 1e-7
    for i in range(len(labels)):
        gold = set(labels[i])
        try:
            pred = set(predictions[i])
        except Exception:
            pred = set()
        tp += len(gold.intersection(pred))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))
    if verbose:
        print('-'*100)
        print(gold, pred)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    micro_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return micro_f1


def calculate_metric_and_errors(task, dataset, df):
    true_labels, pred_labels, ill_formed_idx = extract_labels(task, dataset, df)
    assert len(true_labels) == len(pred_labels)

    label_space = get_label_space(task, dataset)
    if task == "sc":
        # sc use accuracy
        accuracy =  accuracy_score(true_labels, pred_labels)
        metric = accuracy
        metric_name = "accuracy"
    elif task == "MASC" or task=="MSA":
        # sc use accuracy
        accuracy =  accuracy_score(true_labels, pred_labels)
        results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        print(results)
        metric = accuracy
        metric_name = "accuracy"
    elif task == "MRE" :
        if dataset == "MNRE":
            accuracy =  accuracy_score(true_labels, pred_labels)
            results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            print(results)
            metric = accuracy
            metric_name = "accuracy"
    elif task =="MHMR":
        accuracy =  accuracy_score(true_labels, pred_labels)
        results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        print(results)
        metric = accuracy
        metric_name = "accuracy"
    elif task == "MSR":
        accuracy =  accuracy_score(true_labels, pred_labels)
        results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        print(results)
        metric = accuracy
        metric_name = "accuracy"
    elif task =="QA":
        if dataset == "ScienceQA" or dataset == "ScienceQA_no_image" or dataset=="ScienceQA_1" or dataset=="ScienceQA_no_context":
            accuracy =  accuracy_score(true_labels, pred_labels)
            results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            print(results)
            metric = accuracy
            metric_name = "accuracy"
    else:
        raise NotImplementedError

    if task !="QA":
        error_df = df[df["label_text"] != df["prediction"]]
        ill_df = df.iloc[ill_formed_idx]
    else:
        if dataset == "ScienceQA" or dataset == "ScienceQA_no_image" or dataset=="ScienceQA_1" or dataset=="ScienceQA_no_context":
            error_df = df[df["answer"] != df["predictions_index"]]
            ill_df = df.iloc[ill_formed_idx]

    return metric_name, metric, error_df, ill_df, results


def process_file(task, dataset_name, dataset_path):
    print('-'*100)
    
    pred_path = os.path.join(dataset_path, "prediction.csv")

    df = pd.read_csv(pred_path)
    metric_name, metric, error_df, ill_df, results = calculate_metric_and_errors(task, dataset_name, df)
    print(f"{metric_name.title()} score for {dataset_name} = {metric}")

    error_file_path = os.path.join(dataset_path, "error.csv")
    error_df.to_csv(error_file_path, index=False)

    if len(ill_df) > 0:
        print(f"{len(ill_df)} ill-formed outputs")
        ill_file_path = os.path.join(dataset_path,  "ill.csv")
        ill_df.to_csv(ill_file_path, index=False)

    return metric, results


def main():
    args = parse_args()

    setting = args.setting

    shots = args.shots
    seed = args.seed
    model = args.model_name
    prompt_type = args.prompt_type
    retriver_type = args.retriver_type
    test_path = args.test_path
    prediction_path_name = test_path.split('.')[0].split('_')[-1]
    demo_label_prefix_type = args.demo_label_prefix_type
    label_map_type = args.label_map_type
    
    if args.selected_tasks:
        selected_tasks = eval(args.selected_tasks)
    else:
        selected_tasks = ["sc", "mast", "absa", "MSA", "MASC", "MRE", "MHMR", "MSR", "QA"]

    if args.selected_datasets:
        selected_datasets = eval(args.selected_datasets)
    else:
        selected_datasets = None

    for task in selected_tasks:

        if setting in ["zero-shot", "full", "majority", "random"]:
            task_output_folder = f"outputs/{setting}_{args.prompt_type}/model_{args.model_name}/seed_{args.seed}/{task}/"
        elif setting == "few-shot":
            if args.slm_model_name:
                task_output_folder = f"outputs/{args.slm_model_name.split('/')[-1]}/{setting}/shot_{shots}/model_{args.model_name}/seed_{args.seed}/{task}/"
            else:
                task_output_folder = f"outputs/{setting}_{prompt_type}/{retriver_type}/{retriver_type}_{shots}/model_{model}/label_map_{label_map_type}/label_prefix_{demo_label_prefix_type}/seed_{seed}/{task}"
        metric_dict = {}
        results_dict = {}
        for dataset in sorted(os.scandir(task_output_folder), key=lambda e: e.name):
            if dataset.is_dir():
                if selected_datasets is None or dataset.name in selected_datasets:
                    if setting=='few-shot':
                        dataset_path = os.path.join(dataset.path, prediction_path_name)
                    elif setting =='zero-shot':
                         dataset_path = dataset.path
                    os.makedirs(dataset_path, exist_ok=True)
                    metric_dict[dataset.name], results_dict[dataset.name] = process_file(task, dataset.name, dataset_path)

        for dataset in selected_datasets:
            if setting=='few-shot':
                metric_path = os.path.join(task_output_folder, dataset, prediction_path_name, "metric.txt")
            elif setting =='zero-shot':
                metric_path = os.path.join(task_output_folder, dataset, "metric.txt")
            with open(metric_path, 'w') as f:
                for k, v in metric_dict.items():
                    f.write(f"{k}\t{v}\n")
                for k, v in results_dict.items():
                    f.write(f"{k}\t{v}\n")


if __name__ == "__main__":
    main()
