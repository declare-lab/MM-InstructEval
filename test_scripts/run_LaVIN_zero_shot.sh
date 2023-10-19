for dataset in  '["MVSA_Single"]' 
#'["TumEmo"]' #'["MVSA_Multiple"]'
#'["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'
do
    for prompt_type in  "1" 
    do
        for model_type in '13B'
        do
            torchrun --nproc_per_node 1  --master_port='29597' predict.py \
                --ckpt_dir /data/xiaocui/weights \
                --llm_model LLaMA-$model_type \
                --local_rank "3" \
                --tokenizer_path /data/xiaocui/weights/LLaMA-$model_type/tokenizer.model \
                --adapter_path /data/xiaocui/weights/LaVIN/LaVIN-$model_type/sqa-llama-13b-lite.pth \
                --adapter_type attn \
                --selected_tasks '["MSA"]' \
                --selected_datasets $dataset \
                --setting zero-shot \
                --prompt_type $prompt_type \
                --model_name LaVIN_$model_type \
                --model_type "$model_type" \
                --max_seq_len 512 \
                --max_gen_len 128 \
                --test_path test.csv \
                --visual_adapter_type router \
                --root_path multimodal_data
                # --bits 4bit
        done
    done
done


## '["MSR"]': '["Sarcasm"]' 
## '["MHMR"]' : '["hate"]' 
## '["MRE"]': '["MNRE"]'
## '["MSA"]': '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' '["MOSI_2"]' '["MOSI_7"]' '["MOSEI_2"]' '["MOSEI_7"]' 
## '["MABSA"]': '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'