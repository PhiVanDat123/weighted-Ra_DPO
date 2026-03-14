#!/bin/zsh

model_name_1="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/dpo_llama-8b-sft_ultra-feedback_03-10_22-41"
model_name_2="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/dpo_llama-8b-sft_ultra-feedback_reverse_03-11_02-44"
input_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/datasets/ultra-feedback"
output_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/generated-data/ultra-feedback-random"
model1_template="normal"
model2_template="normal"
batch_size=4
num_gpus=4
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python /mnt/data/anhnh2220/dat/weighted-Ra_DPO/tisdpo_weight.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 