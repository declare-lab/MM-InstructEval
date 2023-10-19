for dataset in '["MVSA_Single"]' #'["MOSEI_7"]'
do
    for model_name in 'openflamingo' 
    do 
        for prompt_type in "1" #"1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
        do
            CUDA_VISIBLE_DEVICES=0 python evaluate.py \
            --setting zero-shot \
            --selected_tasks '["MSA"]' \
            --prompt_type $prompt_type \
            --selected_datasets $dataset \
            --model_name $model_name \
            --test_path test.csv \
            --seed 42
        done
    done
done


## '["MSR"]': '["Sarcasm"]' 
## '["MHMR"]' : '["hate"]' 
## '["MRE"]': '["MNRE"]'
## '["MSA"]': '["MVSA_Single"]' '["MVSA_Multiple"]' '["TumEmo"]' '["MOSI_2"]' '["MOSI_7"]' '["MOSEI_2"]' '["MOSEI_7"]' 
## '["MABSA"]': '["Twitter_2015"]' '["Twitter_2017"]' '["MASAD"]'