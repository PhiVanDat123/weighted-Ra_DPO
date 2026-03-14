import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import multiprocessing as mp
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

    # Fix: không dùng model để lấy device nữa
    if device is None:
        device = torch.device("cpu")

    desc = f"GPU-{process_id}" if process_id is not None else "Processing"

    for i in tqdm(range(0, len(prompts_1), batch_size), desc=desc, mininterval=1.0, ncols=80):
        batch_prompts_1 = prompts_1[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]

        batch_weights = []
        for j in range(len(batch_prompts_1)):
            # Tokenize response để biết độ dài
            response_ids = tokenizer.encode(batch_responses[j], add_special_tokens=False)
            response_length = len(response_ids)

            # Sample weight ngẫu nhiên từ U(0,1) cho mỗi token
            weights = [round(float(w), 2) for w in torch.rand(response_length).tolist()]
            batch_weights.append(weights)

        all_weights.extend(batch_weights)

    return all_weights, all_explain_data

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def process_dataset_shard(gpu_id, input_file, model_name_1, model_name_2, model1_template, model2_template, data_shard, batch_size=8):
    # Vẫn set device để log thông tin, nhưng không cần GPU thực sự
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")

    # Chỉ load tokenizer, không load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    tokenizer.pad_token = tokenizer.eos_token

    prompts1 = [promt_dict[model1_template].format(item['prompt']) for item in data_shard]
    prompts2 = [promt_dict[model2_template].format(item['prompt']) for item in data_shard]
    rejected_responses = [item['rejected'] for item in data_shard]
    chosen_responses = [item['chosen'] for item in data_shard]

    print(f"GPU {gpu_id}: Processing {len(data_shard)} examples")

    # Truyền None thay vì model
    rejected_weights, _ = calculate_probability_differences(
        None, None, tokenizer, prompts1, prompts2, rejected_responses,
        batch_size=batch_size, device=device, process_id=gpu_id
    )
    chosen_weights, _ = calculate_probability_differences(
        None, None, tokenizer, prompts1, prompts2, chosen_responses,
        batch_size=batch_size, device=device, process_id=gpu_id
    )

    for i, item in enumerate(data_shard):
        item['rejected_weight'] = rejected_weights[i]
        item['chosen_weight'] = chosen_weights[i]

    return data_shard

def get_output_file(output_dir, file_path):
    file_name = os.path.basename(file_path).split(".")[0]
    output_file = os.path.join(output_dir, f"{file_name}.jsonl")
    return output_file

def parallel_process_file(file_path, args):
    print(f"Processing file: {file_path}")
    data = load_jsonl(file_path)

    # Fix: không bắt buộc phải có GPU nữa vì không dùng model
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus) if available_gpus > 0 else 1

    print(f"Using {num_gpus} workers (available GPUs: {available_gpus})")

    # Split data into shards
    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus
    for i in range(0, len(data), shard_size):
        shards.append(data[i:i+shard_size])

    shards = shards[:num_gpus]
    print(f"Split data into {len(shards)} shards")

    if args.force_sequential or len(shards) == 1:
        print("Using sequential processing")
        results = []
        for i in range(len(shards)):
            result = process_dataset_shard(
                i % max(available_gpus, 1), file_path, args.model_name_1, args.model_name_2,
                args.model1_template, args.model2_template, shards[i], args.batch_size
            )
            results.append(result)
        processed_shards = results
    else:
        print("Using parallel processing with multiprocessing Pool")
        with mp.Pool(num_gpus) as pool:
            results = []
            for i in range(len(shards)):
                result = pool.apply_async(
                    process_dataset_shard,
                    args=(i % max(available_gpus, 1), file_path, args.model_name_1, args.model_name_2,
                          args.model1_template, args.model2_template, shards[i], args.batch_size)
                )
                results.append(result)
            processed_shards = [r.get() for r in results]

    processed_data = []
    for result in processed_shards:
        processed_data.extend(result)

    output_file = get_output_file(args.output_dir, file_path)
    save_jsonl(processed_data, output_file)
    print(f"Saved processed data to {output_file}")

    return output_file

def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("Multiprocessing start method already set, continuing with existing method")

    parser = argparse.ArgumentParser(description="Process dataset with random U(0,1) weights.")
    parser.add_argument('--model_name_1', type=str, required=True,
                        help='Path to the first model (used for tokenizer only).')
    parser.add_argument('--model_name_2', type=str, required=True,
                        help='Path to the second model (unused).')
    parser.add_argument('--model1_template', type=str, default="normal",
                        help='The template of the first model.')
    parser.add_argument('--model2_template', type=str, default="normal",
                        help='The template of the second model.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing JSONL files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing.')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of workers to use for parallel processing.')
    parser.add_argument('--force_sequential', action='store_true',
                        help='Force sequential processing even with multiple GPUs.')

    args = parser.parse_args()

    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} available GPUs")
    # Fix: không raise lỗi khi không có GPU
    if available_gpus > 0 and args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus}.")
        args.num_gpus = available_gpus

    start_time = time.time()
    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]

    processed_files = []
    for file_path in all_files:
        output_file = parallel_process_file(file_path, args)
        processed_files.append(output_file)

    elapsed_time = time.time() - start_time
    print(f"Finished processing all files in {elapsed_time:.2f} seconds")
    print("Processed files:")
    for file in processed_files:
        print(f"  {file}")

if __name__ == "__main__":
    main()