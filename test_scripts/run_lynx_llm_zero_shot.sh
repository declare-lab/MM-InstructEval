#!/bin/bash
for task in '["MSA"]'
do
    for dataset in  '["MOSI_2"]' 
    do
        for prompt_type in "1"
        do
            CUDA_VISIBLE_DEVICES=3 python3 predict.py \
            --model_name lynx_llm \
            --seed 42 \
            --use_context \
            --prompt_type $prompt_type \
            --selected_tasks $task \
            --selected_datasets $dataset \
            --test_path test.csv \
            --model_type 'finetune_lynx'
        done
    done
done



## '["MSR"]': '["Sarcasm"]' 
## '["MHMR"]' : '["hate"]' 
## '["MRE"]': '["MNRE"]'
## '["MSA"]': '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' '["MOSI_2"]' '["MOSI_7"]' '["MOSEI_2"]' '["MOSEI_7"]' 
## '["MABSA"]': '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'