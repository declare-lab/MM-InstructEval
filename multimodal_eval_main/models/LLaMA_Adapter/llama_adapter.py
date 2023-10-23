import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import pandas as pd
import argparse
import random
import requests
from tqdm import tqdm
import numpy as np
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_fixed,
# )
import concurrent.futures
import torch
from PIL import Image
#from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import BytesIO
import cv2
import llama

# from llava.conversation import conv_templates, SeparatorStyle
# from llava.utils import disable_torch_init
# from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
# from llava.model import *
# from llava.model.utils import KeywordsStoppingCriteria

# from lavis.common.config import Config
# from lavis.common.dist_utils import get_rank
# from lavis.common.registry import registry
# from lavis.conversation.conversation import Chat, CONV_VISION
#
# # imports modules for registration
# from lavis.datasets.builders import *
# from lavis.models import *
# from lavis.processors import *
# from lavis.runners import *
# from lavis.tasks import *

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

llama_dir = "/data/xiaocui/weights/LLaMA-7B"
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of samples to use, better under 3")
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random]")
    parser.add_argument("--seed", type=int, default=42, help="[0, 1, 42]")
    parser.add_argument("--shots", type=int, default=-1, help="[1, 5, 10]")
    parser.add_argument('--use_api', action='store_true', help='use api or not')
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks, e.g '[\"sc\"]'")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--ignored_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model_path", type=str, default="/datas/multimodal_LLMs/llava-7b", help="/path/to/model")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned dataset")
    parser.add_argument("--root_path", type=str, default="/datas/multimodal_datasets/multimodal_data-20230629T020002Z-001/multimodal_data", help="the path of multimodal data")
    parser.add_argument("--test_path", type=str, default="test_0_10.csv", help="the path of multimodal data")
    parser.add_argument('--device',type = int, default = 0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--use_context", action="store_true", help="whether use context for ScienceQA")
    return parser.parse_args()

# device = torch.device("cuda:{}").format(args.device)


# Get label space
def get_label_space(task: str, dataset: str) -> list:
    if task == 'MASC':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017":
            label_space = ["positive", "neutral", "negative"]
        elif dataset == 'MASAD':
            label_space = ["positive", "negative"]
    elif task == 'MSC':
        if dataset == 'MVSA_Multiple' or dataset == 'MVSA_Single':
            label_space = ["positive", "neutral", "negative"]
        elif dataset == 'TumEmo':
            label_space = ["angry", "bored", "calm", "fear", "happy", "love", "sad"]
            pass
        elif dataset == 'MOSEI_2' or dataset == "MOSI_2":
            label_space = ["positive", "negative"]
            pass
        elif dataset == "MOSEI_7" or dataset == "MOSI_7":
            label_space = ["positive", "neutral", "negative", "weakly positive", "weakly negative", "strongly positive", "strongly negative"]
    elif task == "MNRE":

        '''
       {'held_on': 18, 'couple': 19, 'member_of': 110, 'alternate_names': 29, 'peer': 156, 'contain': 99, 'nationality': 10, 'subsidiary': 16, 'part_of': 14, 'locate_at': 46, 'place_of_birth': 7, 'present_in': 74, 'charges': 1, 'parent': 4, 'place_of_residence': 29, 'awarded': 4, 'siblings': 1, 'religion': 1, 'neighbor': 2})
        '''
        entity_cat_space = ['loction', 'organization', 'person', 'misc']
        label_space = ['held_on', 'couple', 'member_of', 'alternate_names', 'peer', 'contain', 'nationality',
                       'subsidiary', 'part_of', 'locate_at', 'place_of_birth', 'present_in', 'charges', 'parent',
                       'place_of_residence', 'awarded', 'siblings', 'religion', 'neighbor']

        if "JMNRE" in dataset:
            label_space = (sorted(label_space), sorted(entity_cat_space))
        return label_space
    elif task == "MHM":
        ##Multimodal_Hateful_Memes
        label_space = ["yes", "no"]
    elif task == "Multimodal_Sarcasm_Detection":
        label_space = ["yes", "no"]
    elif task == "Multimodal_Rumor":
        label_space = ['real', "fake"]
    elif task == "QA":
        if dataset == "ScienceQA" or dataset == "ScienceQA_no_image" or dataset == "ScienceQA_1":
            label_space = ["0", '1', "2", "3", "4"]
    else:
        raise NotImplementedError
    return sorted(label_space)


# Function to get the task name and stance target based on the task and dataset
def get_task_name(task: str, dataset: str) -> str:

    if task == 'MASC':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017" or dataset == 'MASAD' :
          task_name = "multimodal aspect-based sentiment classification"
    elif task == 'MSC':
        task_name = "multimodal sentiment classification"
    elif task == "MNRE":
        if 'JMNRE' in dataset:
            task_name = "joint multimodal entity-relation extraction"
        elif dataset == "MRE":
            task_name = "multimodal relation extraction"
    elif task == "MHM":
        task_name = "multimodal hateful detection"
    elif task == "Multimodal_Sarcasm_Detection":
        task_name = "multimodal irony detection"
    elif task=="Multimodal_Rumor":
        task_name = "multimodal fake news detection"
    elif task == "QA":
        task_name ="multimodal question answer"
    else:
        raise NotImplementedError

    return task_name.title()

def generate_fake_data(task, dataset, label_space, row):
    # fake data for dev
    if any(substring in dataset for substring in ["uabsa", "aste", "asqp"]):
        try:
            pred = [random.choice(eval(row["label_text"]))]
        except:
            pred = []
    else:
        pred = str(random.choice(label_space))
    return pred

# Define templates for different tasks and datasets
def generate_template(key, label_space, task_name, **kwargs):
    task_definitions = {
        "MASC": "Given the sentence and the aspect, assign a sentiment label towards \"{target}\" from {label_space}.",
        "MSC": "Given the sentence, assign a sentiment label from {label_space}.",
        "MRE": "Given the sentence, assign a relation label towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\" from {label_space}.",
        # "MRE": "Given the sentence, assign a relation label towards the \"({head_entity}, {head_cat}, {tail_entity}, {tail_cat}\" from {label_space}.",
        "MHM": "Given the sentence, assign a sentiment label from {label_space}.",
        # 'MHM': "Given the sentence, please determine whether or not it contains hate. Assign a sentiment label from {label_space}.",
        "Multimodal_Sarcasm_Detection": "Given the sentence, please determine whether or not it contains irony. Assign a sentiment label from {label_space}.",
        "Multimodal_Rumor": "Given the sentence, please determine whether or not it is fake news. Assign a label from {label_space}.",
        "QA": "Given the question, "
    }

    output_formats = {
        "MASC": "Return label only without any other text.",
        "MSC": "Return label only without any other text.",
        "MRE": "Return label only without any other text.",
        "MHM": "Return label only without any other text.",
        "Multimodal_Sarcasm_Detection": "Return label only without any other text.",
        "Multimodal_Rumor": "Return label only without any other text.",
        "QA": "Return answer only without any other text.",
    }

    if key == "stance":
        task_name += " ({target})".format(**kwargs)

    task_definition = task_definitions[key].format(**kwargs, label_space=label_space)
    output_format = output_formats[key]

    return task_name, task_definition, output_format


# generate demos
def generate_fix_demo(train_df, task, dataset):
    tuple_list = []
    if dataset in ['Twitter_2015', "Twitter_2017", 'MASAD']:
        for i, row in train_df.iterrows():
            aspect = row["aspect"]
            text = row["text"].replace('$T$', aspect)
            label = row["label_text"]
            text += f" (sentiment towards Aspect: \"{aspect}\")"
            image_path = row['image']
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    elif dataset in ['MVSA_Single', "MVSA_Multiple", 'TumEmo']:
        for i, row in train_df.iterrows():
            text = row["text"]
            image_path = row['image']
            label = row["label_text"]
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    elif dataset in ['hate', "Fakeddit", 'MSD']:
        for i, row in train_df.iterrows():
            text = row["text"]
            image_path = row['image']
            label = row["label_text"]
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    elif dataset == "MRE":
        for i, row in train_df.iterrows():
            text = row["text"]
            head_entity = row['head_entity']
            head_cat = row['head_cat']
            tail_entity = row['tail_entity']
            tail_cat = row['tail_cat']
            label = row["label_text"]
            text += f" (relation towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\")"
            image_path = row['image']
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    else:
        sub_df = train_df[['text', 'label_text']]
        tuple_list = [tuple(x) for x in sub_df.to_records(index=False)]
    return tuple_list


# Function to generate prompt for the OpenAI model
def generate_prompt(args, setting, task, dataset, label_space, row, demo_tuples):
    if task != "QA":
        text = row["text"]
        pass
    else:
        text = row['question']
        pass
    if len(text) > 400:
        text = text[:400]
    if task == 'MASC':
        aspect = row['aspect']
    task_name = get_task_name(task, dataset)

    if task == "MASC":
        task_name, task_definition, output_format = generate_template("MASC", label_space, task_name=task_name,
                                                                      target=row["aspect"])
    elif task == "MSC":
        task_name, task_definition, output_format = generate_template("MSC", label_space, task_name=task_name)
    elif task == 'MNRE':
        head_entity = row['head_entity']
        head_cat = row['head_cat']
        tail_entity = row['tail_entity']
        tail_cat = row['tail_cat']
        if dataset == "MRE":
            relation_label_space = label_space
            task_name, task_definition, output_format = generate_template("MRE", relation_label_space,
                                                                          task_name=task_name, head_entity=head_entity,
                                                                          head_cat=head_cat, tail_entity=tail_entity,
                                                                          tail_cat=tail_cat)
        elif dataset == "JMNRE":
            relation_label_space, entity_cat_space = label_space
            task_name, task_definition, output_format = generate_template("JMNRE", relation_label_space,
                                                                          task_name=task_name, head_entity=head_entity,
                                                                          head_cat=head_cat, tail_entity=tail_entity,
                                                                          tail_cat=tail_cat,
                                                                          entity_cat_space=entity_cat_space)
    elif task == "MHM":
        task_name, task_definition, output_format = generate_template("MHM", label_space, task_name=task_name)
    elif task == 'Multimodal_Sarcasm_Detection':
        task_name, task_definition, output_format = generate_template("Multimodal_Sarcasm_Detection", label_space,
                                                                      task_name=task_name)
    elif task == "Multimodal_Rumor":
        task_name, task_definition, output_format = generate_template("Multimodal_Rumor", label_space,
                                                                      task_name=task_name)
        pass
    elif task == "QA":
        ##original_index,question,image,answer,answer_text,choices,hint

        choices = eval(row['choices'])
        task_name = get_task_name(task, dataset)
        task_name, task_definition, output_format = generate_template("QA", label_space='', task_name=task_name)
        option_num = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

        options = ''
        # question = "What is the answer about the above question?"
        for i, choice in enumerate(choices):
            option = option_num[i] + " " + choice + " "
            options += option
        task_definition = task_definition + f"please choose the answer from \"{options}\" to the following question."

    else:
        raise NotImplementedError

    if setting == "zero-shot":
        if args.num_prompts == 1:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                # pass
            # print(prompt)


            if task == 'QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\nLabel:"
                        question = ""
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nLabel:"
                        question = ""
                    pass
                pass
            else:
                prompt = prompt + "Label:"
            pass
        elif args.num_prompts == 2:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                pass
            elif task == "QA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"

            if task == "QA":
                prompt = prompt + "The answer is:"
                pass
            else:
                prompt = prompt + "Question: " + question + "Label:"
            pass
        elif args.num_prompts == 3:
            task_predefinition = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            pre_fix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n"
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
                pass

            if task == 'QA':
                context = row['hint']

                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    # question = "What is the answer about the above question?"
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n Question: {text}\nContext: {context}\n"
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n Question: {text}\n"
                    pass
                prompt = task_predefinition + "### Instruction: \n" + prompt + f"Options: {options}\n" + "### Response:"
                pass
            else:
                prompt = pre_fix + prompt + "Instruction:\n" + question + "Options:\n" + options + "\n" + "### Response:"
            pass
        elif args.num_prompts == 4:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                pass
            if task == 'QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                        question = "What is the answer about the above question?"
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                        question = "What is the answer about the above question?"

            prompt = prompt + "\n" + question
            pass
        elif args.num_prompts == 5:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                options = "".join(
                    ["(" + chr(ord('a') + i) + ")" + " " + x for i, x in zip(range(len(label_space)), label_space)])

            if task == 'QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                    # question = "What is the answer about the above question?"
                prompt = prompt + f"Options: {options}\n" + "The answer is:"
                pass
            else:
                prompt = prompt + "\n" + question + "Options:\n" + options + "\n" + "Answer:"

            pass
        elif args.num_prompts == 6:
            task_predefinition = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            pre_fix = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>"
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"

            if task == 'QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                prompt = task_predefinition + "Human: " + prompt + "AI:"
                pass
            else:
                prompt = pre_fix + "Human:" + prompt + "Human:\n" + question + "AI:"

            pass
        elif args.num_prompts == 7:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
                options = " or ".join(label_space)
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                options = " or ".join(label_space)
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = " or ".join(label_space)
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
                options = " or ".join(label_space)
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = " or ".join(label_space)
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
                options = " or ".join(label_space)

            if task == 'QA':
                context = row['hint']
                choices = eval(row['choices'])
                option_num = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
                new_options = ''
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                        pass
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                    for i, choice in enumerate(choices):
                        choice = "\"" + choice + "\""
                        if i < (len(choices) - 1):
                            option = choice + " or "
                        else:
                            option = choice
                        new_options += option
                prompt = prompt + "Options: " + new_options + "\nThe answer is:"
                pass
            else:
                prompt = prompt + "\n" + question + "Options:\n" + options + "\n" + "Answer:"

            pass
        elif args.num_prompts == 8:
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                pass
            elif task == 'QA':
                context = row['hint']
                if args.use_context:

                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    pass
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"

            prompt = prompt
            pass
        elif args.num_prompts == 9:
            task_predefinition = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
            pre_fix = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n### Instruction:\n"
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text} ### Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on an text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text}\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text}\n"
                question = "whether is the hate about the text-image pair?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"

            if task == 'QA':
                context = row['hint']

                prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n"
                if args.use_context:

                    prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + f"Question: {text}\nContext: {context}\n" + "### Response:"
                    pass
                else:
                    prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + f"Question: {text}\n" + "### Response:"
                pass
            else:
                prompt = pre_fix + prompt + "###Input:\n" + question + "### Response:"

            pass
        elif args.num_prompts == 10:
            pre_fix = "<image>User:"
            if task == "MASC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text} ### Aspect: {aspect}\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text}\n"
            elif task == "MNRE":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text}\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text}\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text}\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Sentence: {text}\n"

            if task == 'QA':
                context = row['hint']
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    pass
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"

                prompt = "User: " + prompt + ":<answer>"
                pass
            else:
                prompt = pre_fix + prompt + "<answer>"

            pass
        
    elif setting == "few-shot":
        demo_string = ""
        for tup in demo_tuples:
            demo_string += f"\nSentence:\n{tup[0]}\nLabel:{tup[1]}\n"
        prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n{demo_string}\nSentence:\n{text}\nLabel:\n"
    else:
        raise NotImplementedError
    return prompt




def process_dataset(args, task, dataset, file_path, output_folder, setting, num_workers, train_path, shots, verbose=False):
    # disable_torch_init()
    df = pd.read_csv(file_path)

    if setting in ["few-shot", "majority"]:
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    print(f"Predict on Task: {task}, Dataset: {dataset}")
    label_space = get_label_space(task, dataset)

    predictions = []
    prompts = []

    prompt_args = []
    if setting in ["zero-shot", "random", "majority"]:
        demo_tuples = None
    elif setting == "few-shot":
        demo_tuples = generate_fix_demo(train_df, task, dataset)
    else:
        raise NotImplementedError

    max_len = 0
    if setting in ["zero-shot", "few-shot"]:
        # cfg = Config(args)
        # vis_processor_cfg = cfg.datasets_cfg.minigpt4_self_instruct_caption.vis_processor.train
        # vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        #
        # model_config = cfg.model_cfg
        # model_config.device_8bit = args.device
        # model_cls = registry.get_model_class(model_config.arch)
        # model = model_cls.from_config(model_config).to('cuda:{}'.format(args.device))
        #
        # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.device))
        device = 'cuda:{}'.format(args.device)
        model, preprocess = llama.load("BIAS-7B", llama_dir, device)
        # print(model.device)

        # model_name = os.path.expanduser(args.model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if "mpt" in model_name.lower():
        #     model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
        #                                                 use_cache=True).cuda(args.device)
        # else:
        #     model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
        #                                                 use_cache=True).cuda(args.device)
        # image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
        #
        # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #
        # count = 0
        # s = 0

        for index, row in tqdm(df.iterrows()):
            # count += 1
            # print('index is {}'.format(index))
            prompt = llama.format_prompt(generate_prompt(args, setting, task, dataset, label_space, row, demo_tuples))
            # prompt = [
            #     '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            #     Human: <image>
            #     Human: Explain why this meme is funny.
            #     AI: ''']
            # print("------------prompt-------------")
            # print(prompt)
            # print("------------prompt-------------")
            # if index==0:
            #     print('prompt is {}'.format((prompt)))
            ##read image
            image_path = row['image']
            raw_image = cv2.imread(image_path)
            raw_image = Image.fromarray(raw_image)
            raw_image = preprocess(raw_image).unsqueeze(0).to(device)
            #image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # chat_state = CONV_VISION.copy()
            # img_list = []
            # chat_state.messages = []
            # llm_message = chat.upload_img(raw_image, chat_state, img_list)
            # chat.ask(prompt, chat_state)
            # outputs = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=300, max_length=2000)[0]
            # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            # print(prompt)
            outputs = model.generate(raw_image,[prompt])[0]
            # if outputs.endswith(stop_str):
            #     outputs = outputs[:-len(stop_str)]
            #     outputs = outputs.strip()
            
            print("sentence is {"+outputs+"}")
            if task == "QA" and args.use_context == False:
                output_path = os.path.join(output_folder, f"prediction-13b-no_context.csv")
                # output_path = os.path.join(output_folder, f"prediction-13b_5.csv")
                # print(output_path, args.use_context)
                pass
            else:
                output_path = os.path.join(output_folder, f"prediction-13b.csv")
                # output_path = os.path.join(output_folder, f"prediction-13b-no_context_5.csv")
                # print(output_path, args.use_context)


            predictions.append(outputs)
            max_len = max(max_len, len(prompt.split()))
            # if index == 0:
            prompt_sample = prompt

            prompt_args.append((prompt))

        for args in prompt_args:
            prompts.append(args[1])

    elif setting in ["random", "majority"]:
        if setting == "majority":
            most_common = train_df["label_text"].value_counts().idxmax()
        for index, row in tqdm(df.iterrows()):
            prompt_sample = ""
            if setting == "random":
                pred = generate_fake_data(task, dataset, label_space, row)
            elif setting == "majority":
                # should use train file
                pred = most_common
            prompts.append("")
            predictions.append(pred)
    else:
        raise NotImplementedError

    # print(f"max_len: {max_len}")
    if verbose:
        print(prompt)
    # df = df.head()
    df["prediction"] = predictions
    # df["prompt"] = prompts

    df.to_csv(output_path, index=False)

    return prompt_sample


# Function to process the task and process datasets
def process_task(args, task, api_key, selected_datasets=None, ignored_datasets=None):

    setting = args.setting
    num_workers = args.num_workers
    shots = args.shots
    seed = args.seed
    root_path = args.root_path
    test_path = args.test_path

    task_folder = os.path.join(root_path, f"{task}")

    if setting in ["zero-shot", "random", "majority"]:
        #output_task_folder = f"outputs/{setting}/model_{model}/seed_{seed}/{task}"
        output_task_folder = "/datas/wangm/multimodal_eval/LLaMA_Adapter/"+str(args.num_prompts)
    elif setting == "few-shot":
        output_task_folder = f"outputs/{setting}/shot_{shots}/model_llava/seed_{seed}/{task}"
    else:
        raise NotImplementedError

    prompt_samples = []
    dataset_names = []

    def check_entry(entry, selected_datasets, ignored_datasets):
        return entry.is_dir() and (selected_datasets is None or entry.name in selected_datasets) \
            and (ignored_datasets is None or entry.name not in ignored_datasets)

    # task_folder /data/multimodal_datasets/Multimodal_Sentiment_Classification
    # 判断task_folder下的数据集是否在selected中，在就加入entries
    entries = (entry for entry in sorted(os.scandir(task_folder), key=lambda e: e.name) if check_entry(entry, selected_datasets, ignored_datasets))
    print(entries)
    for dataset in entries:
        print(dataset)
        output_dataset_folder = os.path.join(output_task_folder, dataset.name, str(args.num_prompts))
        os.makedirs(output_dataset_folder, exist_ok=True)

        file_path = os.path.join(dataset.path, test_path)

        if setting in ["zero-shot", "random"]:
            train_path = None
        elif setting == "majority":
            train_path = os.path.join(f"csv/{task}/{dataset.name}", "train.csv")
        elif setting == "few-shot":
            train_path = os.path.join(dataset.path, f"shot_{shots}", f"seed_{seed}", "train.csv")
        else:
            raise NotImplementedError

        if args.skip_runned:
            pred_file = os.path.join(output_dataset_folder, "prediction.csv")
            if os.path.exists(pred_file):
                print(f"{task} {dataset.name} skiped")
                continue

        prompt_sample = process_dataset(args, task, dataset.name, file_path, output_dataset_folder, setting, num_workers, train_path, shots)

        prompt_samples.append(prompt_sample)
        dataset_names.append(dataset.name)

        prompt_file = os.path.join(output_task_folder, dataset.name, "prompt.txt")
        with open(prompt_file, 'w') as f:
            for task_dataset, prompt in zip(dataset_names, prompt_samples):
                f.write('-'*100+'\n')
                f.write(f"{task}-{task_dataset}:\n{prompt}\n\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(args.selected_tasks)
    selected_tasks = eval(args.selected_tasks) if args.selected_tasks else ["sc", "mast", "absa"]
    selected_datasets = eval(args.selected_datasets) if args.selected_datasets else None
    ignored_datasets = eval(args.ignored_datasets) if args.ignored_datasets else None

    api_key = args.api

    for task in selected_tasks:
        process_task(args, task, api_key, selected_datasets, ignored_datasets)

if __name__ == "__main__":
    main()