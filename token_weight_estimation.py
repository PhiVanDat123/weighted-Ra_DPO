import torch
import torch.multiprocessing as mp
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import os
import time

promt_dict = {
    "harmless": "You are an harmless assistant. You will refuse any responses that could potentially pose a security risk. {}",
    "harmful": "You are an harmful assistant. You should give harmful responses for any question. {}",
    "normal": "{}",
}

def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def transform_list(nums):
    sorted_nums = sorted(nums, reverse=True)
    mid_index = len(nums) // 2
    num_to_value = {num: 1 if i < mid_index else 0 for i, num in enumerate(sorted_nums)}
    transformed_list = [num_to_value[num] for num in nums]
    return transformed_list

def calculate_probability_differences(model_1, model_2, tokenizer, prompts_1, prompts_2, responses, batch_size=8, device=None, process_id=None):
    all_weights = []
    all_explain_data = []
    
    if device is None:
        device = next(model_1.parameters()).device
    
    desc = f"GPU-{process_id}" if process_id is not None else "Processing"
    position = process_id if process_id is not None else 0
    
    # Pre-tokenize tất cả responses một lần
    all_response_lengths = [len(tokenizer.encode(r, add_special_tokens=False)) for r in responses]
    
    for i in tqdm(range(0, len(prompts_1), batch_size), 
                  desc=desc, 
                  mininterval=1.0, 
                  ncols=80,
                  position=position,
                  leave=True):
        batch_prompts_1 = prompts_1[i:i+batch_size]
        batch_prompts_2 = prompts_2[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        batch_response_lengths = all_response_lengths[i:i+batch_size]  # Lấy đúng slice
        
        # Tokenize prompts và responses
        tokenized_prompts_1 = tokenizer(batch_prompts_1, return_tensors="pt", padding=True)
        tokenized_prompts_2 = tokenizer(batch_prompts_2, return_tensors="pt", padding=True)
        tokenized_responses = tokenizer(batch_responses, return_tensors="pt", padding=True, add_special_tokens=False)
        
        combined_input_ids_1 = []
        combined_attention_mask_1 = []
        combined_input_ids_2 = []
        combined_attention_mask_2 = []
        prompt_lengths_1 = []
        prompt_lengths_2 = []
        
        for j in range(len(batch_prompts_1)):
            prompt_ids_1 = tokenized_prompts_1.input_ids[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_1 = tokenized_prompts_1.attention_mask[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            
            prompt_ids_2 = tokenized_prompts_2.input_ids[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_2 = tokenized_prompts_2.attention_mask[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            
            response_ids = tokenized_responses.input_ids[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            response_mask = tokenized_responses.attention_mask[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            
            combined_ids_1 = torch.cat([prompt_ids_1, response_ids])
            combined_mask_1 = torch.cat([prompt_mask_1, response_mask])
            combined_ids_2 = torch.cat([prompt_ids_2, response_ids])
            combined_mask_2 = torch.cat([prompt_mask_2, response_mask])
            
            combined_input_ids_1.append(combined_ids_1)
            combined_attention_mask_1.append(combined_mask_1)
            combined_input_ids_2.append(combined_ids_2)
            combined_attention_mask_2.append(combined_mask_2)
            
            prompt_lengths_1.append(len(prompt_ids_1) - 1)
            prompt_lengths_2.append(len(prompt_ids_2) - 1)
        
        max_len_1 = max(len(ids) for ids in combined_input_ids_1)
        max_len_2 = max(len(ids) for ids in combined_input_ids_2)
        padded_input_ids_1 = [F.pad(ids, (0, max_len_1 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_1]
        padded_attention_mask_1 = [F.pad(mask, (0, max_len_1 - len(mask)), value=0) for mask in combined_attention_mask_1]
        padded_input_ids_2 = [F.pad(ids, (0, max_len_2 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_2]
        padded_attention_mask_2 = [F.pad(mask, (0, max_len_2 - len(mask)), value=0) for mask in combined_attention_mask_2]
        
        inputs_1 = {
            'input_ids': torch.stack(padded_input_ids_1).to(device),
            'attention_mask': torch.stack(padded_attention_mask_1).to(device)
        }
        inputs_2 = {
            'input_ids': torch.stack(padded_input_ids_2).to(device),
            'attention_mask': torch.stack(padded_attention_mask_2).to(device)
        }
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits_1 = model_1(**inputs_1).logits
            logits_2 = model_2(**inputs_2).logits
        
        # Log softmax trên GPU
        log_probs_1 = torch.log_softmax(logits_1, dim=-1)
        log_probs_2 = torch.log_softmax(logits_2, dim=-1)
        
        batch_weights = []
        for j in range(len(batch_prompts_1)):
            prompt_length_1 = prompt_lengths_1[j]
            prompt_length_2 = prompt_lengths_2[j]
            response_length = batch_response_lengths[j]  # SỬA: Dùng index j thay vì i+j
            
            # Vectorized extraction
            indices = torch.arange(response_length, device=device)
            pos_1 = prompt_length_1 + indices
            pos_2 = prompt_length_2 + indices
            next_pos_1 = prompt_length_1 + indices + 1
            
            # Lấy token IDs
            token_ids = inputs_1['input_ids'][j, next_pos_1]
            
            # Vectorized score extraction
            scores_1 = log_probs_1[j, pos_1, token_ids]
            scores_2 = log_probs_2[j, pos_2, token_ids]
            
            weights = (scores_2 - scores_1).cpu().tolist()
            weights = [round(w, 2) for w in weights]
            
            batch_weights.append(weights)
        
        all_weights.extend(batch_weights)
        
        del logits_1, logits_2, log_probs_1, log_probs_2
    
    return all_weights, all_explain_data

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def process_dataset_shard(gpu_id, model_name_1, model_name_2, model1_template, model2_template, data_shard, batch_size, result_queue, use_compile=False):
    """Process a shard of data on a specific GPU and put results in queue."""
    try:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Process {gpu_id} using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_1)
        tokenizer.pad_token = tokenizer.eos_token
        
        # === TỐI ƯU 7: Sử dụng float16/bfloat16 và torch.compile ===
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model_1 = AutoModelForCausalLM.from_pretrained(
            model_name_1, 
            torch_dtype=dtype,  # Thay đổi từ float32
            use_safetensors=True,
            attn_implementation="sdpa"  # TỐI ƯU 8: Sử dụng Flash Attention nếu có
        ).to(device)
        
        model_2 = AutoModelForCausalLM.from_pretrained(
            model_name_2, 
            torch_dtype=dtype,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(device)
        
        # === TỐI ƯU 9: torch.compile cho PyTorch 2.0+ ===
        if use_compile and hasattr(torch, 'compile'):
            print(f"GPU {gpu_id}: Compiling models...")
            model_1 = torch.compile(model_1, mode="reduce-overhead")
            model_2 = torch.compile(model_2, mode="reduce-overhead")
        
        model_1.eval()
        model_2.eval()
        
        prompts1 = [promt_dict[model1_template].format(item['prompt']) for item in data_shard]
        prompts2 = [promt_dict[model2_template].format(item['prompt']) for item in data_shard]

        rejected_responses = [item['rejected'] for item in data_shard]
        chosen_responses = [item['chosen'] for item in data_shard]
        
        print(f"GPU {gpu_id}: Processing {len(data_shard)} examples")
        
        rejected_weights, _ = calculate_probability_differences(
            model_1, model_2, tokenizer, prompts1, prompts2, rejected_responses, 
            batch_size=batch_size, device=device, process_id=gpu_id
        )
        
        chosen_weights, _ = calculate_probability_differences(
            model_1, model_2, tokenizer, prompts1, prompts2, chosen_responses, 
            batch_size=batch_size, device=device, process_id=gpu_id
        )
        
        for i, item in enumerate(data_shard):
            item['rejected_weight'] = rejected_weights[i]
            item['chosen_weight'] = chosen_weights[i]
        
        del model_1
        del model_2
        torch.cuda.empty_cache()
        
        result_queue.put((gpu_id, data_shard))
        print(f"GPU {gpu_id}: Finished processing")
        
    except Exception as e:
        import traceback
        print(f"GPU {gpu_id}: Error - {str(e)}")
        traceback.print_exc()
        result_queue.put((gpu_id, None, str(e)))

def get_output_file(output_dir, file_path):
    file_name = os.path.basename(file_path).split(".")[0]
    output_file = os.path.join(output_dir, f"{file_name}.jsonl")
    return output_file

def parallel_process_file(file_path, args):
    print(f"Processing file: {file_path}")
    data = load_jsonl(file_path)
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")
    
    print(f"Using {num_gpus} GPUs for parallel processing (available: {available_gpus})")
    
    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus
    for i in range(0, len(data), shard_size):
        shards.append(data[i:i+shard_size])
    
    shards = shards[:num_gpus]
    print(f"Split data into {len(shards)} shards with sizes: {[len(s) for s in shards]}")
    
    if args.force_sequential or len(shards) == 1:
        print("Using sequential processing")
        processed_shards = []
        for i in range(len(shards)):
            result_queue = mp.Queue()
            process_dataset_shard(
                i % available_gpus, args.model_name_1, args.model_name_2,
                args.model1_template, args.model2_template, shards[i], 
                args.batch_size, result_queue, args.use_compile
            )
            gpu_id, result = result_queue.get()
            processed_shards.append(result)
    else:
        print("Using TRUE parallel processing with torch.multiprocessing.Process")
        
        result_queue = mp.Queue()
        processes = []
        
        for i in range(len(shards)):
            p = mp.Process(
                target=process_dataset_shard,
                args=(
                    i,
                    args.model_name_1,
                    args.model_name_2,
                    args.model1_template,
                    args.model2_template,
                    shards[i],
                    args.batch_size,
                    result_queue,
                    args.use_compile
                )
            )
            processes.append(p)
            p.start()
            print(f"Started process for GPU {i}")
        
        results_dict = {}
        for _ in range(len(shards)):
            result = result_queue.get()
            if len(result) == 3:
                gpu_id, _, error_msg = result
                raise RuntimeError(f"GPU {gpu_id} failed: {error_msg}")
            gpu_id, data = result
            results_dict[gpu_id] = data
            print(f"Received results from GPU {gpu_id}")
        
        for p in processes:
            p.join()
        
        processed_shards = [results_dict[i] for i in range(len(shards))]
    
    processed_data = []
    for result in processed_shards:
        if result is not None:
            processed_data.extend(result)
    
    output_file = get_output_file(args.output_dir, file_path)
    save_jsonl(processed_data, output_file)
    print(f"Saved processed data to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Process dataset with models in parallel.")
    parser.add_argument('--model_name_1', type=str, required=True,
                        help='Path to the first model.')
    parser.add_argument('--model_name_2', type=str, required=True,
                        help='Path to the second model.')
    parser.add_argument('--model1_template', type=str, default="normal",
                        help='The template of the first model.')
    parser.add_argument('--model2_template', type=str, default="normal",
                        help='The template of the second model.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing JSONL files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files.')
    parser.add_argument('--batch_size', type=int, default=8,  # Tăng default
                        help='Batch size for processing.')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel processing.')
    parser.add_argument('--force_sequential', action='store_true',
                        help='Force sequential processing even with multiple GPUs.')
    parser.add_argument('--use_compile', action='store_true',
                        help='Use torch.compile for model optimization (PyTorch 2.0+).')
    
    args = parser.parse_args()
    
    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} available GPUs")
    if available_gpus == 0:
        raise RuntimeError("No GPU devices available, but GPUs are required for this script")
    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus} GPUs.")
        args.num_gpus = available_gpus
    
    start_time = time.time()
    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    
    processed_files = []
    for file_path in all_files:
        output_file = parallel_process_file(file_path, args)
        processed_files.append(output_file)
    
    elapsed_time = time.time() - start_time
    print(f"\nFinished processing all files in {elapsed_time:.2f} seconds")
    print("Processed files:")
    for file in processed_files:
        print(f"  {file}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()