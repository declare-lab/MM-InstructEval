## Setting model

1. Download weights for each model;
2. Change to your path in the corresponding to the code.

Note: the download weights from huggingface can refer to [utils/download_meta_llama2.py](https://github.com/declare-lab/MM-BigBench/blob/main/utils/download_meta_llama2.py).


### LLaMA-1

Download weights:  

LLaMA-1-7B: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf),
LLaMA-1-13B: [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf)

### LLaMA-2

Download weights:  

LLaMA-2-7B: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf),
LLaMA-2-13B: [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)

### OpenFlamingo

We refer to the official repo [OpenFlamingo](https://github.com/mlfoundations/open_flamingo).

Download weights:  

[OpenFlamingo](https://huggingface.co/openflamingo/OpenFlamingo-9B-deprecated)


### MiniGPT4

We refer to the official repo [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4/).

[MiniGPT4(Vicuna 13B)](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view),
[Vicuna 13B](https://github.com/lm-sys/FastChat).



### mPLUG-Owl

We refer to the official repo  [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl).

Download weights:  

[mplug-owl-llama-7b](https://huggingface.co/MAGAer13/mplug-owl-llama-7b).

### LLaMA-Adapter V2

We refer to the official repo  [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

Download weights:  

[LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main).


### VPGTrans

We refer to the official repo [VPGTrans](https://github.com/VPGTrans/VPGTrans).

Download weights:  

[vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.1).

### LLaVA

We refer to the official repo [LLaVA](https://github.com/haotian-liu/LLaVA). 

Download weights:  

For **LLaVA-7B**, download weights: 

[LLaVA-7b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0). 


For **LLaVA-13B**, download weights: 

[LLaVA-13b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0).

Coverting delta weights refering to the "[Legacy Models (delta weights)](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)".


### Multimodal-GPT

We refer to the official repo [Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT).

Download weights:  

[llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf),
[OpenFlamingo-9B](https://huggingface.co/openflamingo/OpenFlamingo-9B-deprecated),
[mmgpt-lora-v0-release.pt](https://github.com/open-mmlab/Multimodal-GPT#:~:text=LoRA%20Weight%20from-,here,-.).


### LaVIN 

We refer to the official repo [LaVIN](https://github.com/luogen1996/LaVIN).

For **LaVIN-7B**, download weights: 

[LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main),
[sqa-llama-7b-lite.pth](https://drive.google.com/file/d/1oVtoTgt-d9EqmrVic27oZUreN9dLClMo/view).

For **LaVIN-13B**, download weights: 

[LLaMA-13B](https://huggingface.co/TheBloke/llama-13b),
[sqa-llama-13b-lite.pth](https://drive.google.com/file/d/1PyVsap3FnmgXOGXFXjYsAtR75cFypaHw/view).


### Lynx-llm

We refer to the official repo [Lynx-llm](https://github.com/bytedance/lynx-llm/tree/main).

Download weights:  

[EVA01_g_psz14.pt](https://github.com/bytedance/lynx-llm/tree/main#:~:text=eva_vit_1b%20on%20official-,website,-and%20put%20it),
[vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.1),
[finetune_lynx.pt](https://github.com/bytedance/lynx-llm/tree/main#:~:text=pretrain_lynx.pt%20or-,finetune_lynx.pt,-and%20put%20it).


### Flan-T5-XXL, BLIP2, InstructBLIP, Fromage can auto-download.

### Other models will be updated soon.