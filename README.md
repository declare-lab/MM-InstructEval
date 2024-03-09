# MM-BigBench: Evaluating Multimodal Models on Multimodal Content Comprehension Tasks
[Paper](https://arxiv.org/abs/2310.09036) | [Data](https://github.com/declare-lab/MM-BigBench/tree/main/multimodal_data) | [Leaderboard](https://declare-lab.github.io/MM-BigBench/)
<p align="center">
  <img src="Figure/mm-bigbench.png" alt="" width="200" height="300">
</p>

# Why?

The popularity of **multimodal large language models (MLLMs)** has triggered a recent surge in research efforts dedicated to evaluating these models. Nevertheless, existing evaluation studies of MLLMs, such as [MME](https://arxiv.org/abs/2306.13394), [SEED-Bench](https://arxiv.org/abs/2307.16125), [LVLM-eHub](https://arxiv.org/abs/2306.09265), and [MM-Vet](https://arxiv.org/abs/2308.02490), primarily focus on the comprehension and reasoning of unimodal (vision) content, neglecting performance evaluations in the domain of multimodal (vision-language) content understanding. Beyond multimodal reasoning, tasks related to multimodal content comprehension necessitate a profound understanding of multimodal contexts, achieved through the multimodal interaction to obtain a final answer. 

In this project, we introduce a comprehensive assessment framework called **MM-BigBench**, which incorporates a diverse range of metrics to offer an extensive evaluation of the performance of **various models and instructions** across a wide spectrum of diverse **multimodal content comprehension tasks**, including Multimodal Sentiment Analysis (MSA), Multimodal Aspect-Based Sentiment Analysis (MABSA), Multimodal Hateful Memes Recognition (MHMR), Multimodal Sarcasm Recognition (MSR), Multimodal Relation Extraction (MRE), and the Visual Question Answering (VQA) with text context. Consequently, our work complements research on the performance of MLLMs in multimodal comprehension tasks, achieving a more comprehensive and holistic evaluation of MLLMs.

**MM-BigBench**, with a range of diverse metrics to provide a thorough evaluation of different models and instructions, including the Best Performance metric, the Mean Relative Gain metric, the Stability metric, and the Adaptability metric.


## Evaluated Models (14 MLLMs)

|Model Name| Modality   | Model/Code         | Paper         | PLM           | PVM      |ToTal-Paras | Training-Paras |
|----------|------------|---------------------|---------------|---------------|----------|------------|----------------|
|ChatGPT | Text       | [ChatGPT](https://openai.com/blog/chatgpt)                                                                                                                       | [Paper](https://arxiv.org/abs/2303.08774)        | gpt-3.5-turb  | -         | -      | -          | 
|LLaMA1-7B | Text       | [LLaMA-1](https://github.com/facebookresearch/llama/tree/llama_v1) | [Paper](https://arxiv.org/abs/2302.13971) | LLaMA-V1-7B   | -    | 6.74B  | 6.74B |
|LLaMA1-13B| Text       |[LLaMA-1](https://github.com/facebookresearch/llama/tree/llama_v1) | [Paper](https://arxiv.org/abs/2302.13971) | LLaMA-V1-13B | _ | 13.02B | 13.02B |
|LLaMA2-7B  | Text       | [LLaMA-2](https://github.com/facebookresearch/llama) and [llama-recipes](https://github.com/facebookresearch/llama-recipes/)  | [Paper](https://arxiv.org/abs/2307.09288) | LLaMA-V2-7B   | -    | 6.74B  | 6.74B |
|LLaMA2-13B | Text       |[LLaMA-2](https://github.com/facebookresearch/llama) and [llama-recipes](https://github.com/facebookresearch/llama-recipes/)  | [Paper](https://arxiv.org/abs/2307.09288) | LLaMA-V2-13B | _ | 13.02B | 13.02B |
|Flan-T5-XXL | Text |[Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl)  |[Paper](https://arxiv.org/abs/2210.11416)| Flan-T5-XXL | - | 11.14B | 11.14B |
|OpenFlamingo | Multimodal | [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) | [Paper](https://openreview.net/forum?id=EbMuimAbPbs)    | LLaMA-7B | ViT-L/14 | 8.34B | 1.31B |
|Fromage | Multimodal | [Fromage](https://github.com/kohjingyu/fromage) |[Paper](https://dl.acm.org/doi/10.5555/3618408.3619119) | OPT-6.7B | ViT-L/14 | 6.97B | 0.21B |
|LLaVA-7B | Multimodal | [LLaVA-7B](https://github.com/haotian-liu/LLaVA) |[Paper](https://arxiv.org/abs/2304.08485) |LLaMA-7B | ViT-L/14 | 6.74B | 6.74B |
|LLaVA-13B | Multimodal | [LLaVA-7B](https://github.com/haotian-liu/LLaVA) |[Paper](https://arxiv.org/abs/2304.08485) |LLaMA-13B | ViT-L/14 | 13.02B | 13.02B |
|MiniGPT4 | Multimodal | [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) |[Paper](https://arxiv.org/abs/2304.10592) |Vicuna-13B | ViT-g/14 | 14.11B | 0.04B |
|mPLUG-Owl | Multimodal| [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) |[Paper](https://arxiv.org/abs/2304.14178) | LLaMA-7B | ViT-L/14 | 7.12B | 7.12B |
|LLaMA-Adapter V2 | Multimodal | [LLaMA-Adapter V2](https://github.com/ZrrSkywalker/LLaMA-Adapter) | [Paper](https://www.arxiv-vanity.com/papers/2304.15010/) | LLaMA-7B | ViT-L/14 | 7.23B | 7.23B |
|VPGTrans |  Multimodal| [VPGTrans](https://github.com/VPGTrans/VPGTrans) | [Paper](https://arxiv.org/abs/2305.01278) | Vicuna-7B | -  | 7.83B |	0.11B |
|Multimodal-GPT |  Multimodal| [Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT) | [Paper](https://arxiv.org/abs/2305.04790) |  LLaMA-7B | ViT-L-14 | 8.37B | 0.02B |
|LaVIN-7B |  Multimodal| [LaVIN-7B](https://github.com/luogen1996/LaVIN) | [Paper](https://arxiv.org/abs/2305.15023) | LLaMA-7B | ViT-L/14 | 7.17B | 7.17B |
|LaVIN-13B |  Multimodal| [LaVIN-13B](https://github.com/luogen1996/LaVIN) | [Paper](https://arxiv.org/abs/2305.15023) | LLaMA-13B | ViT-L/14 | 13.36B | 13.36B |
| Lynx |  Multimodal| [Lynx](https://github.com/bytedance/lynx-llm) | [Paper](https://arxiv.org/abs/2307.02469) | Vicuna-7B |Eva-ViT-1b | 8.41B | 0.69B |
|BLIP-2 |Multimodal|[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [Paper](https://arxiv.org/abs/2301.12597) | FlanT5-XXL | ViT-g/14 | 12.23B | 0.11B |
|InstructBLIP | Multimodal|[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip#instructblip-towards-general-purpose-vision-language-models-with-instruction-tuning) | [Paper](https://arxiv.org/abs/2305.06500) | FlanT5-XXL | ViT-g/14 | 12.31B | 0.45B |


Note: Refer to [Setting Models](multimodal_eval_main/models/README.md) for more information.

### Results

For detailed results, please go to our [MM-BigBench leaderboard](https://declare-lab.github.io/MM-BigBench/)

## Setup

Install dependencies and download data.

```
conda create -n mm-bigbench python=3.8 -y
conda activate mm-bigbench
pip install -r requirements.txt
```

## Fast Start

### Select evaluated dataset, model, prompt_type

1. Select the evaluated task and dataset in the "*.sh" file;
2. Select the model to be evaluated in the "*.sh" file;
3. Select the prompt type, from "1" to "10" in the "*.sh" file;
4. Run the corresponding "*.sh" file.


### Running the inference of difffernt models

For ChatGPT, LLaMA-V1-7B, LLaMA-V1-13B, LLaMA-V2-7B, LLaMA-V2-13B, Text-FlanT5-XXL, BLIP-2, InstructBLIP, Fromage, OpenFlamingo, Multimodal-GPT,  mPLUG-Owl, MiniGPT4, LLaMA-Adapterv2, VPGTrans, LLaVA-7B, LLaVA-13 models:
```
sh test_scripts/run_scripts.sh

## Change the "model_name" in the ".sh" file to corresponding to the 'chatgpt', 'decapoda-llama-7b-hf', 'decapoda-llamab-hf', 'meta-llama2-7b-hf', 'meta-llama2-13b-hf', 'text_flan-t5-xxl', 'blip2_t5', 'blip2_instruct_flant5xxl', 'fromage', 'openflamingo', 'mmgpt', 'mplug_owl', 'minigpt4', 'llama_adapterv2', 'vpgtrans', 'llava_7b', 'llava_13b'.


```

For LaVIN model: 
```
sh test_scripts/run_LaVIN_zero_shot.sh
```

For Lynx model:
```
sh test_scripts/run_lynx_llm_zero_shot.sh
```


### Eval the results to get the accuracy metric

```
sh test_scripts/eval_scripts.sh
```

## Diverse Metrics

Metrics used in our paper can be found in [Evaluation Metrics](evaluation).


## Citation

```bibtex
@inproceedings{Yang2023MMBigBenchEM,
  title={MM-BigBench: Evaluating Multimodal Models on Multimodal Content Comprehension Tasks},
  author={Xiaocui Yang and Wenfang Wu and Shi Feng and Ming Wang and Daling Wang and Yang Li and Qi Sun and Yifei Zhang and Xiaoming Fu and Soujanya Poria},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:264127863}
}
```
