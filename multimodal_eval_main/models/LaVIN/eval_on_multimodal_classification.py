# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import pandas as pd
import argparse
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter
from util.base_prompt import build_prompt
from util.base_prompt_for_multimodal_classification import multimodal_sentiment_classification_promopt
from dataclasses import dataclass
import re
import random

import warnings
import pandas as pd
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from util.apply_delta import apply_model_delta_online

warnings.filterwarnings('ignore')

# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of samples to use, better under 3")
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random]")
    parser.add_argument("--seed", type=int, default=42, help="[0, 1, 42]")
    parser.add_argument("--shots", type=int, default=-1, help="[1, 5, 10]")
    parser.add_argument('--use_api', action='store_true', help='use api or not')
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--task", type=str, default=None, help="list of string of tasks, e.g '[\"sc\"]'")
    parser.add_argument("--dataset_name", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--ignored_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model_name", type=str, default="LLaMA-v1", help="[blip2_t5, blip2_vicuna_instruct, instructblip]")
    parser.add_argument("--model_type", type=str, default="7B", help="[pretrain_flant5xxl, vicuna7b, flant5xxl]")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned dataset")
    parser.add_argument("--data_root", type=str, default="/data/xiaocui/code/Multimodal_LLMs/LLM-Sentiment/multimodal_data", help="the path of multimodal data")
    parser.add_argument("--test_path", type=str, default="test_0_10.csv", help="the path of multimodal data")
    parser.add_argument("--prompt_type", type=str, default="1", help="the type of prompt")
    parser.add_argument("--ckpt_dir", type=str, default="/data/xiaocui/weights/LLaMA-7B", help="the path of checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="/data/xiaocui/weights/LLaMA-7B/tokenizer.model", help="the path of tokenizer")
    parser.add_argument("--adapter_path", type=str, default='/data/xiaocui/code/Multimodal_LLMs/LaVIN/weights/sqa-llama-7b.pth', help="the path of tokenizer")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--max_seq_len", type=int, default=512, help="")
    parser.add_argument("--max_gen_len", type=int, default=512, help="")
    parser.add_argument("--max_batch_size", type=int, default=1, help="")
    parser.add_argument("--local_rank", type=str, default="1", help="")
    return parser.parse_args()
args = parse_args()



@dataclass
class PromptArgs:
    prompt_format='QCM-ALE'
    use_caption=True
    options=["A", "B", "C", "D", "E"]

# os.environ['RANK'] = str(0)
os.environ["LOCAL_RANK"]=args.local_rank
# os.environ["WORLD_SIZE"]='1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29510'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
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



def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))
    num = len(results)
    assert num == 4241

    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():

        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    scores = {
        'acc_natural':
        get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
        get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
        get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
        get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
        get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
        get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)

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
    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))





def main(
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
    
):
    ckpt_dir = args.ckpt_dir
    tokenizer_path = args.tokenizer_path
    adapter_path=args.adapter_path
    data_root = args.data_root
    max_seq_len = args.max_seq_len
    
    task = args.task
    dataset_name = args.dataset_name
    setting = args.setting
    model_name = args.model_name
    model_type = args.model_type
    
    
    print(max_batch_size,max_seq_len)
    print('use caption: ',use_caption)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir,llm_model, tokenizer_path, adapter_path, local_rank, world_size , max_seq_len, max_batch_size,
        adapter_type,adapter_dim,adapter_scale,hidden_proj,visual_adapter_type,
    temperature,use_vicuna)

    print('split: ', split)
    data_path = os.path.join(data_root, task, dataset_name, f"{split}_0_10.csv")
    all_lines =[]
    data_df = pd.read_csv(data_path)
    for index, row in data_df.iterrows():
        all_lines.append(row)

    total_items=len(all_lines) 
    print('total_items: ',total_items)

    image_transforms=transforms.Compose(
                                        [transforms.Resize((224, 224), 
                                        interpolation=Image.BICUBIC),
                                         transforms.ToTensor(), 
                                         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    prompt_args=PromptArgs()
    # prompt_args.prompt_format = prompt_format
    # prompt_args.use_caption = use_caption
    # prompt_args.options = options

    pattern = re.compile(r'The answer is [A-Za-z].')

    answers = []
    preds=[]
    for i in range(total_items//max_batch_size+1):
        print('progresses: ',i,' / ', total_items//max_batch_size+1)
        batch_json_lines = all_lines[i*max_batch_size:(i+1)*max_batch_size]
        if len(batch_json_lines)==0:
            break
        indicators = []
        prompts=[]
        images = []
        for json_line in batch_json_lines:
            prompt, output, image_path, answer = multimodal_sentiment_classification_promopt(json_line, task=task, dataset=dataset_name, setting=setting)
            if image_path is not None:
                image = Image.open(image_path).convert('RGB')
                image = image_transforms(image)
                indicator = 1
            else:
                image = torch.Tensor(torch.zeros(3, 224, 224).float())
                indicator = 0
            prompts.append(prompt)
            answers.append(answer)
            images.append(image.unsqueeze(0))
            indicators.append(indicator)
        images=torch.cat(images,0)
        print(images.shape)


        results = generator.generate(
            prompts,images=images,indicators=indicators, max_gen_len=args.max_gen_len, temperature=generation_temperature, top_p=top_p,n_feats=n_prompt
        )

        for result in results:
            print(result)
            # pred = pattern.findall(result)

            # if len(pred) >= 1:
            #     pred = pred[0]  # 'A', 'B', ...
            # else:
            #     print(result)
            #     pred = "FAILED"
            preds.append(result)
    print("++++++++++++++++++++++++prompt+++++++++++++++++++++++++++")
    print(prompts[0])
    #evaluations
    results={}
    correct=0
    data_df['predictions_original'] = preds
    # df["prompt"] = prompts

    output_dir = os.path.join('/data/xiaocui/code/Multimodal_LLMs/LaVIN/outputs', setting, f'model_{model_name}_{model_type}', task, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prediction.csv")
    data_df.to_csv(output_path, index=False)
    
    for i, prediction in enumerate(preds):
        if answers[i]==prediction:
            correct += 1
        # results[batch_json_lines[i][json_line['original']]] = prediction
    acc = correct / len(preds) * 100
    print('overall accuracy: ', acc)

    with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
        f.write(prompts[0])
    
    with open(os.path.join(output_dir, 'preds.json'), 'w') as f:
        json.dump(results,f)

    # scores=get_scores('./preds.json',os.path.join(data_root, 'problems.json'))
    # print(scores)
    # import time
    # with open(str(time.time())+'.txt','w') as f:
    #     f.write(str(scores))


if __name__ == "__main__":
    main()
    ## torchrun --nproc_per_node 1 --master_port='24998' eval_on_multimodal_classification.py
