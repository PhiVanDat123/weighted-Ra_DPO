import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Optional
import argparse
import torch.nn.functional as F
import os
import multiprocessing as mp
import time

# ──────────────────────────────────────────────────────────────────────────────
# System prompt templates
# ──────────────────────────────────────────────────────────────────────────────

def extract_prompt_text(prompt_str: str) -> str:
    text = prompt_str.strip()
    if text.startswith("Human:"):
        text = text[len("Human:"):].strip()
    if text.endswith("\n\nAssistant:"):
        text = text[:-len("\n\nAssistant:")].strip()
    return text

SYSTEM_PROMPTS = {
    "harmless": "You are a harmless assistant. You will refuse any responses that could potentially pose a security risk.",
    "harmful":  "You are a harmful assistant. You should give harmful responses for any question.",
    "normal":   None,
}

'''
def build_prompt(tokenizer, user_content: str, system_prompt: Optional[str]) -> str:
    if "<|start_header_id|>" not in user_content and "<|im_start|>" not in user_content:
        user_content = extract_prompt_text(user_content)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}"
        return user_content

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
'''

# ──────────────────────────────────────────────────────────────────────────────
# KL-based gradient attribution  (single backward pass — fast)
# ──────────────────────────────────────────────────────────────────────────────

def compute_kl_gradient_attribution(
    dpo_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    device: Optional[torch.device] = None,
) -> torch.FloatTensor:
    """
    Compute per-token importance scores via a SINGLE backward pass.

    Instead of L separate backward passes (one per position, extremely slow),
    we compute the sum of all per-position KL values and backprop once:

        KL(t) = D_KL( pi_dpo(·|context_t) || pi_ref(·|context_t) )
        loss  = sum_t KL(t)                     # scalar
        I_t   = || d(loss) / d(e_t) ||_1        # accumulated gradient

    The gradient at e_t accumulates influence from all downstream positions
    where token t was part of the context — a richer signal than isolating
    each position separately, and O(1) backward passes instead of O(L).

    Args:
        dpo_model : frozen DPO checkpoint (pi_dpo), no grad
        ref_model : reference model (pi_ref), gradient flows through embeddings
        input_ids : (B, L) token ids
        attention_mask : (B, L) binary mask, 1 for real tokens
        device    : target device

    Returns:
        importances : (B, L) float tensor, 0 at pad positions.
    """
    if device is None:
        device = input_ids.device

    input_ids      = input_ids.to(device)
    attention_mask = (
        attention_mask.to(device)
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.long)
    )

    original_ref_training = ref_model.training
    ref_model.eval()
    dpo_model.eval()

    try:
        # ── pi_ref embeddings — single tensor grad flows through ──────────────
        ref_embeddings = (
            ref_model.get_input_embeddings()(input_ids)
            .detach()
            .requires_grad_(True)
        )  # (B, L, D)

        # ── pi_dpo logits — fully detached, no grad ───────────────────────────
        with torch.no_grad():
            dpo_probs = F.softmax(
                dpo_model(input_ids=input_ids, attention_mask=attention_mask).logits,
                dim=-1,
            )  # (B, L, V)

        # ── pi_ref logits — through grad-enabled embedding path ───────────────
        ref_logits = ref_model(
            inputs_embeds=ref_embeddings,
            attention_mask=attention_mask,
        ).logits  # (B, L, V)

        # ── Per-position KL: (B, L) ───────────────────────────────────────────
        # F.kl_div(log_q, p) = sum_v p*(log p - log q)  [per element]
        # sum over vocab dim → scalar KL per (batch, position)
        kl_all = F.kl_div(
            input=F.log_softmax(ref_logits, dim=-1),  # log pi_ref, grad path
            target=dpo_probs,                          # pi_dpo,    no grad
            reduction="none",
        ).sum(dim=-1)  # (B, L)

        # Zero out padding positions before summing
        total_kl = (kl_all * attention_mask.float()).sum()  # scalar

        # ── Single backward pass ──────────────────────────────────────────────
        grads = torch.autograd.grad(
            outputs=total_kl,
            inputs=ref_embeddings,
            retain_graph=False,
            create_graph=False,
        )[0]  # (B, L, D)

        # L1 norm over embedding dim → importance per token
        #importances = grads.abs().sum(dim=-1).float()   # (B, L)
        importances = grads.norm(p=2, dim=-1).float()
        importances = importances * attention_mask.float()

        return importances

    finally:
        ref_model.train(original_ref_training)


# ──────────────────────────────────────────────────────────────────────────────
# Token-level weight computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_token_importance_weights(
    dpo_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    response_start_positions: list,
    device: Optional[torch.device] = None,
) -> list:
    """
    Return per-response-token importance weights (one list per sample in batch).

    Pipeline:
      1. KL-gradient attribution over full (prompt + response) sequence.
      2. Extract importances for response tokens only.
      3. Normalize: w_t = I_norm  (pure KL-gradient signal).
      4. Scale so mean = 1 over the response span (sum = n).
    """
    if device is None:
        device = input_ids.device

    batch_size = input_ids.shape[0]

    with torch.enable_grad():
        importances = compute_kl_gradient_attribution(
            dpo_model=dpo_model,
            ref_model=ref_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
        )  # (B, L)

    all_weights = []

    for i in range(batch_size):
        resp_start = response_start_positions[i]

        valid_mask = attention_mask[i].to(torch.bool).clone()
        valid_mask[:resp_start] = False
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        n = int(valid_idx.numel())
        if n == 0:
            all_weights.append([])
            continue

        scores    = importances[i][valid_idx].to(torch.float32).clamp_min(0)
        score_sum = scores.sum()

        if score_sum > 0:
            weights = scores / score_sum
        else:
            # Fallback: uniform if all gradients are zero
            weights = torch.ones(n, device=device, dtype=torch.float32) / n

        # Scale so mean = 1 (sum = n) — preserves loss magnitude
        weights = weights * float(n)

        all_weights.append([round(float(w), 4) for w in weights.tolist()])

    return all_weights


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def calculate_importance_weights(
    dpo_model,
    ref_model,
    tokenizer,
    prompts: list,
    responses: list,
    batch_size: int = 4,
    device=None,
    process_id=None,
) -> list:
    """Compute KL-gradient importance weights for a list of (prompt, response) pairs."""
    all_weights = []

    if device is None:
        device = next(dpo_model.parameters()).device

    desc = f"GPU-{process_id}" if process_id is not None else "Processing"

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc, mininterval=1.0, ncols=80):
        batch_prompts   = prompts[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]

        tokenized_prompts = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokenized_responses = tokenizer(
            batch_responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        combined_ids    = []
        combined_masks  = []
        response_starts = []

        for j in range(len(batch_prompts)):
            p_mask_bool = tokenized_prompts.input_ids[j] != tokenizer.pad_token_id
            p_ids  = tokenized_prompts.input_ids[j][p_mask_bool]
            p_mask = tokenized_prompts.attention_mask[j][p_mask_bool]

            r_mask_bool = tokenized_responses.input_ids[j] != tokenizer.pad_token_id
            r_ids  = tokenized_responses.input_ids[j][r_mask_bool]
            r_mask = tokenized_responses.attention_mask[j][r_mask_bool]

            response_starts.append(int(len(p_ids)))
            combined_ids.append(torch.cat([p_ids,  r_ids]))
            combined_masks.append(torch.cat([p_mask, r_mask]))

        max_len      = max(len(x) for x in combined_ids)
        padded_ids   = [F.pad(x, (0, max_len - len(x)), value=tokenizer.pad_token_id) for x in combined_ids]
        padded_masks = [F.pad(x, (0, max_len - len(x)), value=0) for x in combined_masks]

        inputs = {
            "input_ids":      torch.stack(padded_ids).to(device),
            "attention_mask": torch.stack(padded_masks).to(device),
        }

        batch_weights = compute_token_importance_weights(
            dpo_model=dpo_model,
            ref_model=ref_model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            response_start_positions=response_starts,
            device=device,
        )
        all_weights.extend(batch_weights)

    return all_weights


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU shard processing
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset_shard(
    gpu_id: int,
    input_file: str,
    dpo_model_name: str,
    ref_model_name: str,
    model_template: str,
    data_shard: list,
    batch_size: int = 4,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"GPU {gpu_id}: loading models on {device}")

    tokenizer = AutoTokenizer.from_pretrained(dpo_model_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # pi_dpo: frozen, no grad
    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    dpo_model.eval()
    dpo_model.config.use_cache = False
    for p in dpo_model.parameters():
        p.requires_grad_(False)

    # pi_ref: grad flows through embeddings via inputs_embeds
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    ref_model.eval()
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad_(False)
    '''
    system_prompt = SYSTEM_PROMPTS[model_template]
    prompts = [
        build_prompt(tokenizer, item["prompt"], system_prompt)
        for item in data_shard
    ]
    '''
    prompts = [item["prompt"] for item in data_shard]
    rejected_responses = [item["rejected"] for item in data_shard]
    chosen_responses   = [item["chosen"]   for item in data_shard]

    print(f"GPU {gpu_id}: processing {len(data_shard)} examples")

    rejected_weights = calculate_importance_weights(
        dpo_model, ref_model, tokenizer, prompts, rejected_responses,
        batch_size=batch_size, device=device, process_id=gpu_id,
    )
    chosen_weights = calculate_importance_weights(
        dpo_model, ref_model, tokenizer, prompts, chosen_responses,
        batch_size=batch_size, device=device, process_id=gpu_id,
    )

    for i, item in enumerate(data_shard):
        item["rejected_weight"] = rejected_weights[i]
        item["chosen_weight"]   = chosen_weights[i]

    del dpo_model, ref_model
    torch.cuda.empty_cache()
    return data_shard


def get_output_file(output_dir: str, file_path: str) -> str:
    file_name = os.path.basename(file_path).split(".")[0]
    return os.path.join(output_dir, f"{file_name}.jsonl")


def parallel_process_file(file_path: str, args) -> str:
    print(f"Processing file: {file_path}")
    data = load_jsonl(file_path)

    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")

    shard_size = (len(data) + num_gpus - 1) // num_gpus
    shards = [data[i : i + shard_size] for i in range(0, len(data), shard_size)][:num_gpus]
    print(f"Using {num_gpus} GPU(s), {len(shards)} shard(s)")

    if args.force_sequential or len(shards) == 1:
        processed_shards = [
            process_dataset_shard(
                i % available_gpus, file_path,
                args.dpo_model_name, args.ref_model_name, args.model_template,
                shards[i], args.batch_size,
            )
            for i in range(len(shards))
        ]
    else:
        with mp.Pool(num_gpus) as pool:
            results = [
                pool.apply_async(
                    process_dataset_shard,
                    args=(
                        i % available_gpus, file_path,
                        args.dpo_model_name, args.ref_model_name, args.model_template,
                        shards[i], args.batch_size,
                    ),
                )
                for i in range(len(shards))
            ]
            processed_shards = [r.get() for r in results]

    processed_data = [item for shard in processed_shards for item in shard]
    output_file = get_output_file(args.output_dir, file_path)
    save_jsonl(processed_data, output_file)
    print(f"Saved → {output_file}")
    return output_file


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="KL-divergence gradient-attribution token weights for Ra-DPO datasets."
    )
    parser.add_argument("--dpo_model_name", type=str, required=True,
                        help="Frozen DPO checkpoint (pi_dpo). Provides target KL distribution.")
    parser.add_argument("--ref_model_name", type=str, required=True,
                        help="Reference model (pi_ref). Gradient flows through its embeddings.")
    parser.add_argument("--model_template", type=str, default="normal",
                        choices=list(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--input_dir",      type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True)
    parser.add_argument("--batch_size",     type=int, default=2,
                        help="Two models loaded per GPU — keep low if VRAM is tight.")
    parser.add_argument("--num_gpus",       type=int, default=8)
    parser.add_argument("--force_sequential", action="store_true")

    args = parser.parse_args()

    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} available GPU(s)")
    if available_gpus == 0:
        raise RuntimeError("No GPU devices available")
    args.num_gpus = min(args.num_gpus, available_gpus)

    start_time = time.time()
    all_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".jsonl")
    )

    print(f"Files to process: {all_files}")
    processed_files = [parallel_process_file(fp, args) for fp in all_files]

    print(f"\nFinished in {time.time() - start_time:.2f}s")
    for f in processed_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()