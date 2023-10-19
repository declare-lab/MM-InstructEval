for dataset in  '["TumEmo"]'  #'["MASAD"]'
#'["TumEmo"]' #'["MVSA_Multiple"]'
#'["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'
do
    for prompt_type in "9" "10"
    do
        torchrun --nproc_per_node 1 --master_port='29531' predict_LaVIN_four.py \
        --ckpt_dir /data/xiaocui/weights/LLaMA-7B \
        --local_rank "1" \
        --tokenizer_path /data/xiaocui/weights/LLaMA-7B/tokenizer.model \
        --selected_tasks '["MSC"]' \
        --selected_datasets $dataset \
        --setting zero-shot \
        --prompt_type $prompt_type \
        --model_name LaVIN \
        --model_type "7B" \
        --max_seq_len 512 \
        --max_gen_len 512 \
        --test_path test.csv
        # --use_api \
    done
done


## Multimodal_Sarcasm_Detection: MSD
## Multimodal_Rumor: Fakeddit
## MHM: hate
## MNRE: MRE
## MSC: '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' 
## MASC: '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'\\
#torchrun --nproc_per_node 1 --master_port='29501'
  # --master_port='29501' python -m torch.distributed.launch predict_llamav1.py \