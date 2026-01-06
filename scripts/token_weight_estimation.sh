#!/bin/zsh

model_name_1=$MODEL_NAME_1
model_name_2=$MODEL_NAME_2
input_dir="/mnt/datpv/DPO/TIS-DPO/datasets/hh-helpful"
output_dir="/mnt/datpv/DPO/TIS-DPO/generated-data/hh-v1"
model1_template="normal"
model2_template="normal"
batch_size=2
num_gpus=2
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python /mnt/datpv/DPO/TIS-DPO/token_weight_estimation.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 