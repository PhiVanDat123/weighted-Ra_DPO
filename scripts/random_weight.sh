#!/bin/zsh

model_name_1="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/qwen3-sft"
model_name_2="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/qwen3-sft"
input_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/datasets/ultra-feedback"
output_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/generated-data/ultra-feedback-random"
model1_template="normal"
model2_template="normal"
batch_size=64   # Tăng lên vì không cần lo VRAM, chỉ tokenize thôi
num_gpus=1      # Không cần multi-GPU nữa
force_sequential=true  # Dùng sequential vì không load model, multiprocessing không cần thiết

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the processing script
python /mnt/data/anhnh2220/dat/weighted-Ra_DPO/random_weight.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi)