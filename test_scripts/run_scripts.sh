for dataset in '["ScienceQA"]'  #'["MASAD"]'
do
    for model_name in 'llava_7b' #'minigpt4' 'llama_adapterv2' 'vpgtrans'
    ## 'decapoda-llama-7b-hf' 'decapoda-llamab-hf' 'meta-llama2-7b-hf' 'meta-llama2-13b-hf' 'text_flan-t5-xxl' 
    ## 'blip2_t5' 'blip2_instruct_flant5xxl' 'fromage' 'openflamingo' 'mmgpt' 'mplug_owl' 'minigpt4' 'llama_adapterv2' 'vpgtrans' 'llava_7b' 'llava_13b'
    do 
        for prompt_type in  "1"
        do 
            CUDA_VISIBLE_DEVICES=0 python predict.py \
            --selected_tasks '["QA"]' \
            --selected_datasets $dataset \
            --setting zero-shot \
            --prompt_type $prompt_type \
            --model_name $model_name \
            --max_output_new_length 100 \
            --test_path test_0_10.csv \
            --root_path multimodal_data \
            --llama_path_for_llama_adapter /data/xiaocui/weights/LLaMA-7B \
            --cfg_path /data/xiaocui/code/MMBigBench-github/multimodal_eval_main/models/VPGTrans/lavis/projects/blip2/demo/vl_vicuna_demo.yaml \
            --llava_model_path /data/xiaocui/weights/llava-7b
        done
    done
done

   
## --mplug_owl_pretrained_ckpt /data/xiaocui/weights/mplug/MAGAer13/mplug-owl-llama-7b \ 
## --open_flamingo_path /data/xiaocui/weights/openflamingo/OpenFlamingo-9B/checkpoint.pt \
## --finetune_path  /data/xiaocui/weights/Multimodal_GPT/mmgpt-lora-v0-release.pt
## for llama-v1, llama-v2
## --llama_path /data/xiaocui/weights/meta-llama/Llama-2-7b-hf \
## --llama_path /data/xiaocui/weights/decapoda-research/llama-7b-hf \
## for minigpt4
## --cfg_path multimodal_eval_main/models/MiniGPT4/eval_configs/minigpt4_eval.yaml \
## --minigpt4_pretrained_ckpt /data/xiaocui/weights/MiniGPT4/pretrained_minigpt4.pth
## for vpgtrans
## --cfg_path /data/xiaocui/code/MMBigBench/multimodal_eval_main/models/VPGTrans/lavis/projects/blip2/demo/vl_vicuna_demo.yaml

## for llava_7b
#  --llama_model_path /data/xiaocui/weights/llava-7b

## for llava_13b
#  --llama_model_path /data/xiaocui/weights/llava-13b

## '["MSR"]': '["Sarcasm"]' 
## '["MHMR"]' : '["hate"]' 
## '["MRE"]': '["MNRE"]'
## '["MSA"]': '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' '["MOSI_2"]' '["MOSI_7"]' '["MOSEI_2"]' '["MOSEI_7"]' 
## '["MABSA"]': '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'
## '["QA"]': '["ScienceQA"]'