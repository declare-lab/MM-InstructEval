for dataset in '["MSD"]' #'["Twitter_2017"]' '["MASAD"]'
do
    for prompt_type in  "3" '5' '7' #"4" "5" "6" "7" "8" "9" "10"
    do
        CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --setting zero-shot \
        --selected_tasks '["Multimodal_Sarcasm_Detection"]' \
        --selected_datasets $dataset \
        --setting zero-shot \
        --prompt_type $prompt_type \
        --model_name LaVIN \
        --model_type "7B" 
    done
done
## Multimodal_Sarcasm_Detection: MSD
## Multimodal_Rumor: Fakeddit
## MHM: hate
## MNRE: MRE
## MSC: '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' 
## MASC: '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'\\