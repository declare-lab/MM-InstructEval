# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter
import random
import fairscale.nn.model_parallel.initialize as fs_init
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.apply_delta import apply_model_delta_online

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
    parser.add_argument("--model_name", type=str, default="LLaMA-v1", help="[blip2_t5, blip2_vicuna_instruct, instructblip]")
    parser.add_argument("--model_type", type=str, default="7B", help="[pretrain_flant5xxl, vicuna7b, flant5xxl]")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned dataset")
    parser.add_argument("--root_path", type=str, default="/data/xiaocui/code/Multimodal_LLMs/LLM-Sentiment/multimodal_data", help="the path of multimodal data")
    parser.add_argument("--test_path", type=str, default="test_0_10.csv", help="the path of multimodal data")
    parser.add_argument("--prompt_type", type=str, default="1", help="the type of prompt")
    parser.add_argument("--ckpt_dir", type=str, default="/datas/multimodal_LLMs/LLaMA-7B", help="the path of checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="/datas/multimodal_LLMs/LLaMA-7B/tokenizer.model", help="the path of tokenizer")
    parser.add_argument("--adapter_path", type=str, default='/data/xiaocui/code/Multimodal_LLMs/LaVIN/weights/sqa-llama-7b.pth', help="the path of tokenizer")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--max_seq_len", type=int, default=512, help="")
    parser.add_argument("--max_gen_len", type=int, default=512, help="")
    parser.add_argument("--max_batch_size", type=int, default=1, help="")
    parser.add_argument("--local_rank", type=str, default="1", help="")
    return parser.parse_args()
args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


os.environ["LOCAL_RANK"]=args.local_rank
# os.environ["WORLD_SIZE"]='1'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29518'

def setup_model_parallel() -> Tuple[int, int]:
    dist.init_process_group(backend='nccl', init_method='env://',rank=0)
    local_rank = int(os.environ.get("LOCAL_RANK", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print('+++++++++++++++++++++local_rank is {}+++++++++++'.format(local_rank))
    print('+++++++++++++++++++++WORLD_SIZE is {}+++++++++++'.format(world_size))
    # torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(42)
    return local_rank, world_size

def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path)  / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        print(llama_model_path + model_name + '/consolidated.00.pth')
        checkpoint = torch.load(llama_model_path  + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location="cpu")
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params
    
def load(
    ckpt_dir: str,
    llm_model:str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim:int,
    adapter_scale:float,
    hidden_proj:int,
    visual_adapter_type: str,
    temperature: float,
    use_vicuna: bool
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")


    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)
    set_Clip_Adapter(model.backbone.visual, visual_adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)

    torch.set_default_tensor_type(torch.FloatTensor)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    parameters = get_parameter_number(model)
    print("+++++++++++++++++++++++++++++++++++++++++++=") 
    print(parameters)
    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

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
    elif task == "MNRE":
        
        '''
       {'held_on': 18, 'couple': 19, 'member_of': 110, 'alternate_names': 29, 'peer': 156, 'contain': 99, 'nationality': 10, 'subsidiary': 16, 'part_of': 14, 'locate_at': 46, 'place_of_birth': 7, 'present_in': 74, 'charges': 1, 'parent': 4, 'place_of_residence': 29, 'awarded': 4, 'siblings': 1, 'religion': 1, 'neighbor': 2})
        '''
        entity_cat_space = ['loction', 'organization', 'person', 'misc']
        label_space = ['held_on', 'couple', 'member_of', 'alternate_names', 'peer', 'contain', 'nationality', 'subsidiary', 'part_of', 'locate_at', 'place_of_birth', 'present_in', 'charges', 'parent', 'place_of_residence', 'awarded', 'siblings', 'religion', 'neighbor']
        
        if "JMNRE" in dataset:
            label_space = (sorted(label_space), sorted(entity_cat_space))
        return label_space
    elif task== "MHM":
        ##Multimodal_Hateful_Memes
        label_space = ["yes", "no"]
    elif task=="Multimodal_Sarcasm_Detection":
        label_space = ["yes", "no"]
    elif task=="Multimodal_Rumor":
        label_space = ['real', "fake"]
    
    else:
        raise NotImplementedError
    return sorted(label_space)


# Function to get the task name and stance target based on the task and dataset
def get_task_name(task: str, dataset: str) -> str:

    if task == 'MASC':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017" or dataset == 'MASAD' :
          task_name = "multimodal aspect-based sentiment classification"
    elif task == 'MSC':
        if dataset == 'MVSA_Multiple' or dataset == 'MVSA_Single' or dataset == 'TumEmo' :
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
        "MASC": "Given the text-image pair and the aspect, assign a sentiment label towards \"{target}\" from {label_space}.",
        "MSC": "Given the text-image pair, assign a sentiment label from {label_space}.",
        "MRE": "Given the text-image pair, assign a relation label towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\" from {label_space}.",
        # "MRE": "Given the text-image pair, assign a relation label towards the \"({head_entity}, {head_cat}, {tail_entity}, {tail_cat}\" from {label_space}.",
        # "MHM": "Given the text-image pair, assign a sentiment label from {label_space}.",
        'MHM': "Given the text-image pair, please determine whether or not it contains hate. Assign a sentiment label from {label_space}.",
        "Multimodal_Sarcasm_Detection": "Given the text-image pair, please determine whether or not it contains irony. Assign a sentiment label from {label_space}.",
        "Multimodal_Rumor": "Given the text-image pair, please determine whether or not it is fake news. Assign a label from {label_space}.",
    }

    output_formats = {
        "MASC": "Return label only without any other text.",
        "MSC": "Return label only without any other text.",
        "MRE": "Return label only without any other text.",
        "MHM": "Return label only without any other text.",
        "Multimodal_Sarcasm_Detection": "Return label only without any other text.",
        "Multimodal_Rumor": "Return label only without any other text.",
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
def generate_prompt(setting, task, dataset, label_space, row, demo_tuples,  prompt_type):
    text = row["text"]
    text = ' '.join(text.split(' ')[:200])
    if task == 'MASC':
        aspect = row['aspect']
    task_name = get_task_name(task, dataset)

    if task == "MASC":
        task_name, task_definition, output_format = generate_template("MASC", label_space, task_name=task_name, target=row["aspect"])
    elif task == "MSC":
        task_name, task_definition, output_format = generate_template("MSC", label_space, task_name=task_name)
    elif task=='MNRE':
        head_entity = row['head_entity']
        head_cat = row['head_cat']
        tail_entity = row['tail_entity']
        tail_cat = row['tail_cat']
        if dataset == "MRE":
            relation_label_space = label_space
            task_name, task_definition, output_format = generate_template("MRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat)
        elif dataset == "JMNRE":
            relation_label_space, entity_cat_space = label_space
            task_name, task_definition, output_format = generate_template("JMNRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat, entity_cat_space=entity_cat_space)
    elif task == "MHM":
        task_name, task_definition, output_format = generate_template("MHM", label_space, task_name=task_name)
    elif task =='Multimodal_Sarcasm_Detection':
        task_name, task_definition, output_format = generate_template("Multimodal_Sarcasm_Detection", label_space, task_name=task_name)
    elif task=="Multimodal_Rumor":
        task_name, task_definition, output_format = generate_template("Multimodal_Rumor", label_space, task_name=task_name)    
    else:
        raise NotImplementedError

    if setting == "zero-shot":
        # if task=="MASC":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
        #     question = "what is the sentiment about the aspect based on an text-image pair?\n"
        # elif task == "MSC":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
        #     question = "what is the sentiment about the text-image pair? \n"
        #     options = "neutral or negative or positive \n"
        # elif task == "MNRE":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
        #     question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
        # elif task == "MHM":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
        #     question = "whether or not the text-image pair contains the hateful memes?\n"
        # elif task == "Multimodal_Sarcasm_Detection":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
        #     question = "whether or not the text-image pair contains irony?\n"
        # elif task == "Multimodal_Rumor":
        #     prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
        #     question = "whether or not the text-image pair is the fake news?\n"    
            
        # prompt = prompt + "Question: " + question + "Answer:"
        if prompt_type == "1":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\nLabel:"
                question = ""
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                prompt = prompt+"Label:"
            else:
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Label:"
        
        elif prompt_type == "2":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or notthe text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
            prompt = prompt + "Question: " + question + "Answer:"
            
        elif prompt_type == "3":
            task_predefinition = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "Is the image-text pair true news or fake news?\n"
                options = "(a) true (b) fake"
                
            prompt = task_predefinition + "### Instruction: \n" + prompt + "### Instruction: \n" + question + f"Options: {options}\n" + "### Response:"
            
        elif prompt_type == "4":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
            prompt = prompt + question 
            
        elif prompt_type == "5":
            
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                # question = "Is the image-text pair real news or fake news?\n"
                question = "whether or not the text-image pair is the fake news?\n"
                options = "(a) real (b) fake"
                
            prompt = prompt + "Question: " + question  + f"Options: {options}\n" + "Answer:"
            
        elif prompt_type == "6":
            task_predefinition = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            
            if task == "MSC":        
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task=="MASC":   
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text} Aspect: {aspect} \n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nHuman: {text}\n"
                # question = "whether or not the text-image is the fake news?\n"
                question = "Is the image-text pair real news or fake news?"
                
            
            prompt = task_predefinition + "Human: " + prompt + "Human: " + question + "AI:"
            
        elif prompt_type == "7":
            if task == "MSC":        
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset == 'MVSA_Single' or dataset == 'MVSA_Multiple':
                    options = "neutral or negative or positive \n"
                else:
                    options = "angry or bored or calm or fear or happy or love or sad \n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task=="MASC":   
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect} \n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset!= 'MASAD':
                    options = 'neutral or negative or positive \n'
                else:
                    options = 'negative or positive \n'
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options =  "held_on or couple or member_of or alternate_names or peer or contain or nationality or subsidiary or part_of or locate_at or place_of_birth or present_in or charges or parent or place_of_residence or awarded or siblings or religion or neighbor\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
                options = "yes or no \n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "yes or no \n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                # question = "whether or not the text-image is the fake news?\n"
                question = "Is the image-text pair real news or fake news?"
                options = 'real or fake \n'
            prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
                
        elif prompt_type == "8":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"              
            prompt = prompt       
        
        elif prompt_type == "9":
            task_predefinition = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\n### Input:\n {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "Is the image-text pair true news or fake news?\n"
                options = "(a) true (b) fake"
                
            prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + question + "### Response:"
    
        elif prompt_type == "10":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSequence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "Is the image-text pair true news or fake news?\n"
                options = "(a) true (b) fake"
                
            prompt = "User: " + prompt + "Question: " + question + ":<answer>"
        elif prompt_type == "11":
            if task=="MASC":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
            elif task == "MSC":
                prompt = f"Please perform {task_name} task. Sentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MNRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
            elif task == "MHM":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hateful memes?\n"
            elif task == "Multimodal_Sarcasm_Detection":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task == "Multimodal_Rumor":
                prompt = f"Please perform {task_name} task. \n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair is the fake news?\n"
            prompt = prompt + question 
    elif setting == "few-shot":
        demo_string = ""
        for tup in demo_tuples:
            # image_path = tup[-1]
            # image_prompt = 'Write a detailed description for the image.'
            # image_input= MultimodalSequence(
            #                             parts=[
            #                                 MultimodalPart(content=image_path, is_image=True),
            #                                 MultimodalPart(content=image_prompt),
            #                             ]
            #                         )
            # image_output = model.run(image_input)
            image_description = tup[-1]
            
            demo_string += f"\nImage Description: {image_description}\nSentence: {tup[0]}\nLabel:{tup[1]}\n"
       
        # if task!="MNRE":
        #     for tup in demo_tuples:
        #         demo_string += f"\nSentence:\n{tup[0]}\nLabel:{tup[1]}\n"
        # else:
        #     for tup in demo_tuples:
        #         demo_string += f"\nSentence:\n{tup[0]}\nRelation:{tup[1]}\n"
        if task=="MASC":   
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text} Aspect: {aspect}\n"
            question = "what is the sentiment about the aspect based on the text-image pair?\n"
        elif task == "MSC":        
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "what is the sentiment about the text-image pair?\n"
        elif task == "MNRE":
            head_entity = row['head_entity']
            head_cat = row['head_cat']
            tail_entity = row['tail_entity']
            tail_cat = row['tail_cat']
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n{demo_string}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}."
            question = "what has relation between the head entity and the tail entity about the text-image pair?\n"
        elif task == "MHM":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains the hateful memes?\n"
        elif task == "Multimodal_Sarcasm_Detection":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains irony?\n"
        elif task == "Multimodal_Rumor":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            # question = "whether or not the text-image is the fake news?\n"
            question = "Is the image-text pair real news or fake news?"
            
        prompt = prompt + "Question: " + question + "Answer:" 
        
        
    else:
        raise NotImplementedError
    return prompt

def process_dataset(
                    task, dataset, file_path, output_folder, model_name, model_type, setting, num_workers, train_path, shots, 
                    ckpt_dir, tokenizer_path, adapter_path, max_seq_len, max_gen_len,
                    max_batch_size=1,
                    llm_model:str='7B',
                    generation_temperature: float = 0.1,
                    top_p: float = 0.75,
                    split='test',
                    prompt_format='CQM-A',
                    use_caption=False,
                    options=["poritive", "negative", "neutral"],
                    adapter_type='repattn',
                    adapter_dim=8,
                    adapter_scale=1,
                    n_prompt=6,
                    hidden_proj=128,
                    visual_adapter_type='normal',
                    temperature=5.,
                    use_vicuna=False,
                    verbose=False, args=None):
    df = pd.read_csv(file_path)

    if setting in ["few-shot", "majority"]:
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    print(f"Predict on Task: {task}, Dataset: {dataset}")
    label_space = get_label_space(task, dataset)

    predictions = []
    predictions_original = []
    prompts = []

    prompt_args = []
    if setting in ["zero-shot", "random", "majority"]:
        demo_tuples = None
    elif setting == "few-shot":
        demo_tuples = generate_fix_demo(train_df, task, dataset)
    else:
        raise NotImplementedError

    max_len = 0
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    image_transforms=transforms.Compose(
                                        [transforms.Resize((224, 224), 
                                        interpolation=Image.BICUBIC),
                                         transforms.ToTensor(), 
                                         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    
    if setting in ["zero-shot", "few-shot"]:
        if model_name is not None:
            generator = load(
                             ckpt_dir,llm_model, tokenizer_path, adapter_path, local_rank, world_size , max_seq_len, max_batch_size,
                             adapter_type,adapter_dim,adapter_scale,hidden_proj,visual_adapter_type,
                             temperature,use_vicuna)
            for index, row in tqdm(df.iterrows()):
                # print('index is {}'.format(index))
                images = []
                prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples, args.prompt_type)
                if index==0:
                    print('prompt is {}'.format((prompt)))
                ##read image
                image_path = row['image']
                if image_path is not None:
                    image = Image.open(image_path).convert('RGB')
                    image = image_transforms(image)
                    indicator = 1
                else:
                    image = torch.Tensor(torch.zeros(3, 224, 224).float())
                    indicator = 0
                images.append(image.unsqueeze(0))
                images=torch.cat(images,0)
                pred =  generator.generate(
                                            prompts=[prompt],
                                            images=images,
                                            indicators=[indicator],
                                            max_gen_len=max_gen_len, temperature=generation_temperature, top_p=top_p,
                                            n_feats=n_prompt)
                pred=pred[0]
                
                if args.prompt_type=='1':
                    str1= 'Label:'
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="2" or args.prompt_type=="5" or args.prompt_type=="7":
                    str1="Answer:"
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="3" or args.prompt_type=="9" or args.prompt_type=="11":
                    str1="### Response:"
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="4":
                    if task == 'MSC':
                        str1 = "what is the sentiment about the text-image pair?"
                    elif task=='MASC':
                        str1 = 'what is the sentiment about the aspect based on the text-image pair?'
                    elif task == "MHM":
                        str1 = "whether or not the text-image pair contains the hateful memes?"
                    elif task == "Multimodal_Sarcasm_Detection":
                        str1 = "whether or not the text-image pair contains irony?"
                    elif task=="MNRE":
                        str1="what has relation between the head entity and the tail entity about the text-image pair?"
                    index=pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="6":
                    str1="AI:"
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="8":
                    str1="Sentence:"
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                elif args.prompt_type=="10":
                    str1="<answer>"
                    index = pred.find(str1)
                    pred_original = pred[index:].lower()
                if args.prompt_type =="3" or args.prompt_type =="5" or args.prompt_type =="9":
                    if dataset == "MVSA_Multiple" or dataset == "MVSA_Single" or dataset == "Twitter_2015" or dataset=="Twitter_2017":
                        if "(a)" in pred_original:
                            pred_original = "neutral "
                        elif "(b)" in pred_original:
                            pred_original = "negative "
                        elif "(c)" in pred_original:
                            pred_original="positive "
                    elif dataset == "TumEmo":
                        ##Options: (a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad
                        if "(a)" in pred_original:
                            pred_original = "angry "
                        elif "(b)" in pred_original:
                            pred_original = "bored "
                        elif "(c)" in pred_original:
                            pred_original="calm "
                        elif "(d)" in pred_original:
                            pred_original = "fear "
                        elif "(e)" in pred_original:
                            pred_original = "happy "
                        elif "(f)" in pred_original:
                            pred_original = "love "
                        elif "(g)" in pred_original:
                            pred_original = "sad "
                    elif dataset == "MASAD":
                        # options = "(a) negative (b) positive"
                        if "(a)" in pred_original:
                            pred_original = "negative "
                        elif "(b)" in pred_original:
                            pred_original="positive "
                    elif dataset == "MSD" or dataset=="hate":
                        if "(a)" in pred_original:
                            pred_original = "yes "
                        elif "(b)" in pred_original:
                            pred_original="no "


                predictions_original.append(pred_original)
                flag_set = set()
                flag=0
                for label in label_space:
                    if label in pred_original:
                        flag_set.add(label)
                if len(flag_set)==1: 
                    pred= flag_set.pop()    
                else:
                    pred= 'nan'
                
                predictions.append(pred)
                max_len = max(max_len, len(prompt.split()))
                # if index == 0:
                prompt_sample = prompt

                prompt_args.append((model_type, prompt))

            for args in prompt_args:
                prompts.append(args[1])
        else:
            for index, row in tqdm(df.iterrows()):
                prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples)
                max_len = max(max_len, len(prompt.split()))
                if index == 0:
                    prompt_sample = prompt
                pred = generate_fake_data(task, dataset, label_space, row)
                prompts.append(prompt)
                predictions.append(pred)
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
    df["prediction"] = predictions
    df['predictions_original'] = predictions_original
    # df["prompt"] = prompts

    output_path = os.path.join(output_folder, f"prediction.csv")
    df.to_csv(output_path, index=False)

    return prompt_sample


# Function to process the task and process datasets
def process_task(args, task, api_key, selected_datasets=None, ignored_datasets=None):

    setting = args.setting
    num_workers = args.num_workers
    shots = args.shots
    seed = args.seed
    model = args.model_name
    root_path = args.root_path
    test_path = args.test_path
    model_type = args.model_type
    prompt_type = args.prompt_type
    ckpt_dir=args.ckpt_dir
    tokenizer_path=args.tokenizer_path
    temperature=args.temperature
    top_p=args.top_p
    max_seq_len=args.max_seq_len
    max_gen_len=args.max_gen_len
    max_batch_size=args.max_batch_size
    adapter_path = args.adapter_path

    task_folder = os.path.join(root_path, f"{task}")

    if setting in ["zero-shot", "random", "majority"]:
        output_task_folder = f"outputs/{setting}_{prompt_type}/model_{model}/seed_{seed}/{task}"
    elif setting == "few-shot":
        output_task_folder = f"outputs/{setting}/shot_{shots}/model_{model}/seed_{seed}/{task}"
    else:
        raise NotImplementedError

    prompt_samples = []
    dataset_names = []

    def check_entry(entry, selected_datasets, ignored_datasets):
        return entry.is_dir() and (selected_datasets is None or entry.name in selected_datasets) \
            and (ignored_datasets is None or entry.name not in ignored_datasets)

    entries = (entry for entry in sorted(os.scandir(task_folder), key=lambda e: e.name) if check_entry(entry, selected_datasets, ignored_datasets))
    for dataset in entries:
        output_dataset_folder = os.path.join(output_task_folder, dataset.name)
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

        prompt_sample = process_dataset(task, dataset.name, file_path,          
                                        output_dataset_folder, model, model_type, setting, num_workers, train_path, shots,ckpt_dir, tokenizer_path, adapter_path, max_seq_len, max_gen_len, args=args)

        prompt_samples.append(prompt_sample)
        dataset_names.append(dataset.name)

    prompt_file = os.path.join(output_task_folder, dataset.name, "prompt.txt")
    with open(prompt_file, 'w') as f:
        for task_dataset, prompt in zip(dataset_names, prompt_samples):
            f.write('-'*100+'\n')
            f.write(f"{task}-{task_dataset}:\n{prompt}\n\n")


def main():
    # args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(args.selected_tasks)
    print(f"++++++++++++++++++The seed is {args.seed} and the prompt_type is {args.prompt_type}+++++++++++++++++++++++++++++++++++++++++++++")
    selected_tasks = eval(args.selected_tasks) if args.selected_tasks else ["sc", "mast", "absa"]
    selected_datasets = eval(args.selected_datasets) if args.selected_datasets else None
    ignored_datasets = eval(args.ignored_datasets) if args.ignored_datasets else None

    api_key = args.api

    for task in selected_tasks:
        process_task(args, task, api_key, selected_datasets, ignored_datasets)

if __name__ == "__main__":
    main()



