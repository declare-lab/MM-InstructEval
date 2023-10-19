for dataset in '["MOSI_2"]'  #'["MASAD"]'
do
    for model_name in 'text_flan-t5-xxl' ## 'text_flan-t5-xxl' 'blip2_t5' 'blip2_instruct_flant5xxl' 'fromage' 'openflamingo' 'mmgpt'
    do 
        for prompt_type in  "1"
        do 
            CUDA_VISIBLE_DEVICES=0 python predict.py \
            --selected_tasks '["MSA"]' \
            --selected_datasets $dataset \
            --setting zero-shot \
            --prompt_type $prompt_type \
            --model_name $model_name \
            --max_output_new_length 10 \
            --test_path test_0_10.csv \
            --root_path multimodal_data \
            # --open_flamingo_path /data/xiaocui/weights/openflamingo/OpenFlamingo-9B/checkpoint.pt \
            # --llama_path /data/xiaocui/weights/decapoda-research/llama-7b-hf \
            # --finetune_path  /data/xiaocui/weights/Multimodal_GPT/mmgpt-lora-v0-release.pt
        done
    done
done

## '["MSR"]': '["Sarcasm"]' 
## '["MHMR"]' : '["hate"]' 
## '["MRE"]': '["MNRE"]'
## '["MSA"]': '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' '["MOSI_2"]' '["MOSI_7"]' '["MOSEI_2"]' '["MOSEI_7"]' 
## '["MABSA"]': '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'