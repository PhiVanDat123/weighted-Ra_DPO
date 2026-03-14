#!/bin/zsh

# ─────────────────────────────────────────────────────────────────────────────
# dpo_model_name  → frozen DPO checkpoint (π_dpo)
#                   Provides the TARGET distribution for KL computation.
#                   • Round 1 iterative DPO : DPO model from round N-1
#                   • Single-round          : your trained DPO checkpoint
#
# ref_model_name  → reference model (π_ref)
#                   Gradient flows through this model's embeddings.
#                   • Typically the SFT / base model used as reference
#                     during DPO training — must match what the DPO trainer uses.
# ─────────────────────────────────────────────────────────────────────────────
dpo_model_name="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/qwen3-dpo"   # ← frozen DPO checkpoint
ref_model_name="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/output/qwen3-sft"   # ← reference / SFT model

input_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/datasets/ultra-feedback"
output_dir="/mnt/data/anhnh2220/dat/weighted-Ra_DPO/generated-data/ultra-feedback-l2"

# "normal" → no system prompt, prompt is passed through the model's chat template only.
# Must match what your DPO trainer uses.
model_template="normal"

batch_size=1         # keep low — two models are loaded per GPU
num_gpus=4
force_sequential=false
#lambda_importance=0.8      
#prior_sigma_div=4         

mkdir -p "$output_dir"

python /mnt/data/anhnh2220/dat/TIS-DPO/token_weight_estimation.py \
  --dpo_model_name    "$dpo_model_name"    \
  --ref_model_name    "$ref_model_name"    \
  --model_template    "$model_template"    \
  --input_dir         "$input_dir"         \
  --output_dir        "$output_dir"        \
  --batch_size        "$batch_size"        \
  --num_gpus          "$num_gpus"          \
  $([ "$force_sequential" = true ] && echo "--force_sequential")