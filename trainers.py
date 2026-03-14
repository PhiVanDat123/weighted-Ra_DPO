import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

def _tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                          average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)


def tdpo_loss(chosen_logps_margin: torch.FloatTensor,
              rejected_logps_margin: torch.FloatTensor,
              chosen_position_kl: torch.FloatTensor,
              rejected_position_kl: torch.FloatTensor,
              beta: float, alpha: float = 0.5, if_tdpo2: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the TDPO loss for a batch of policy and reference model log probabilities."""
    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)    # tdpo1
    else:
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def tisdpo_loss(chosen_logps_margin: torch.FloatTensor,
                rejected_logps_margin: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                beta: float, alpha: float = 0.5, token_level: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def radpo_loss(chosen_logps_margin: torch.FloatTensor,
               rejected_logps_margin: torch.FloatTensor,
               chosen_position_risk_ratio: torch.FloatTensor,
               rejected_position_risk_ratio: torch.FloatTensor,
               beta: float, alpha: float = 0.5, if_radpo2: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the Ra-DPO loss for a batch of policy and reference model log probabilities."""
    chosen_values = chosen_logps_margin + chosen_position_risk_ratio
    rejected_values = rejected_logps_margin + rejected_position_risk_ratio

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_radpo2:
        # Ra-DPO1
        logits = chosen_rejected_logps_margin - (rejected_position_risk_ratio - chosen_position_risk_ratio)
    else:
        # Ra-DPO2
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_risk_ratio - chosen_position_risk_ratio.detach())

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def _radpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None,
                           confidence_level: float = 0.5, is_split_risk_ratio: bool = True,
                           is_cal_risk_distribution_logps: bool = False, average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities/risk ratio of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    distribution_logps = logits.float().log_softmax(-1)
    reference_distribution_ps = reference_logits.float().softmax(-1)
    reference_distribution_logps = reference_distribution_ps.log()

    per_position_kl = (reference_distribution_ps * (reference_distribution_logps - distribution_logps)).sum(-1)

    if not is_cal_risk_distribution_logps:
        per_position_risk_ratio = _calculate_cvar(reference_distribution_logps, distribution_logps,
                                                   reference_distribution_ps, confidence_level,
                                                   is_split_risk_ratio).sum(-1)
    else:
        per_position_risk_ratio, _, _ = _cal_risk_distribution_logps(reference_distribution_logps, distribution_logps,
                                                                      reference_distribution_ps, confidence_level,
                                                                      is_split_risk_ratio)
        per_position_risk_ratio = per_position_risk_ratio.sum(-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()

    if average_log_prob:
        return (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_risk_ratio * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * weights * loss_mask).sum(-1), \
            (per_position_kl * weights * loss_mask).sum(-1), \
            (per_position_risk_ratio * weights * loss_mask).sum(-1), \
            (per_token_logps * weights * loss_mask).sum(-1)


def _calculate_cvar(reference_distribution, distribution, probabilities, confidence_level, is_split_risk_ratio=True):
    """Calculate CVaR (Conditional Value at Risk) for Ra-DPO."""
    distribution = reference_distribution - distribution

    if is_split_risk_ratio:
        mid_index = distribution.size(2) // 2

        chosen_distribution, rejected_distribution = distribution[:, :, :mid_index], distribution[:, :, mid_index:]
        chosen_probabilities, rejected_probabilities = probabilities[:, :, :mid_index], probabilities[:, :, mid_index:]

        chosen_VaR = torch.quantile(chosen_distribution, 1-confidence_level, dim=2).unsqueeze(-1)
        rejected_VaR = torch.quantile(rejected_distribution, 1-confidence_level, dim=2).unsqueeze(-1)

        chosen_mask = chosen_distribution > chosen_VaR
        rejected_mask = rejected_distribution > rejected_VaR

        chosen_weighted_losses_above_VaR = chosen_probabilities * chosen_distribution * chosen_mask.float()
        rejected_weighted_losses_above_VaR = rejected_probabilities * rejected_distribution * rejected_mask.float()

        CVaR = torch.cat((chosen_weighted_losses_above_VaR, rejected_weighted_losses_above_VaR), dim=2)
    else:
        VaR = torch.quantile(distribution, 1-confidence_level, dim=2).unsqueeze(-1)
        mask = distribution > VaR
        weighted_losses_above_VaR = probabilities * distribution * mask.float()
        CVaR = weighted_losses_above_VaR

    return CVaR


def _cal_risk_distribution_logps(reference_distribution_logps, distribution_logps, probabilities, confidence_level, is_split_risk_ratio=True):
    """Calculate risk distribution log probabilities for Ra-DPO."""
    if is_split_risk_ratio:
        mid_index = distribution_logps.size(2) // 2

        chosen_reference_distribution_logps, rejected_reference_distribution_logps = reference_distribution_logps[:, :, :mid_index], reference_distribution_logps[:, :, mid_index:]
        chosen_distribution_logps, rejected_distribution_logps = distribution_logps[:, :, :mid_index], distribution_logps[:, :, mid_index:]

        chosen_reference_distribution_logps_quantile = torch.quantile(chosen_reference_distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)
        chosen_distribution_logps_quantile = torch.quantile(chosen_distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)
        rejected_reference_distribution_logps_quantile = torch.quantile(rejected_reference_distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)
        rejected_distribution_logps_quantile = torch.quantile(rejected_distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)

        chosen_reference_distribution_logps_mask = chosen_reference_distribution_logps > chosen_reference_distribution_logps_quantile
        chosen_distribution_logps_mask = chosen_distribution_logps > chosen_distribution_logps_quantile
        rejected_reference_distribution_logps_mask = rejected_reference_distribution_logps > rejected_reference_distribution_logps_quantile
        rejected_distribution_logps_mask = rejected_distribution_logps > rejected_distribution_logps_quantile

        chosen_reference_distribution_logps_VaR = chosen_reference_distribution_logps_quantile * chosen_reference_distribution_logps_mask.float()
        chosen_distribution_logps_VaR = chosen_distribution_logps_quantile * chosen_distribution_logps_mask.float()
        rejected_reference_distribution_logps_VaR = rejected_reference_distribution_logps_quantile * rejected_reference_distribution_logps_mask.float()
        rejected_distribution_logps_VaR = rejected_distribution_logps_quantile * rejected_distribution_logps_mask.float()

        chosen_distribution = chosen_reference_distribution_logps_VaR - chosen_distribution_logps_VaR
        rejected_distribution = rejected_reference_distribution_logps_VaR - rejected_distribution_logps_VaR

        reference_distribution_logps_risk = torch.cat((chosen_reference_distribution_logps_VaR, rejected_reference_distribution_logps_VaR), dim=2)
        distribution_logps_risk = torch.cat((chosen_distribution_logps_VaR, rejected_distribution_logps_VaR), dim=2)

        chosen_probabilities, rejected_probabilities = probabilities[:, :, :mid_index], probabilities[:, :, mid_index:]

        chosen_weighted_losses_above_VaR = chosen_probabilities * chosen_distribution
        rejected_weighted_losses_above_VaR = rejected_probabilities * rejected_distribution

        CVaR = torch.cat((chosen_weighted_losses_above_VaR, rejected_weighted_losses_above_VaR), dim=2)
    else:
        reference_distribution_logps_quantile = torch.quantile(reference_distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)
        distribution_logps_quantile = torch.quantile(distribution_logps, 1-confidence_level, dim=2).unsqueeze(-1)

        reference_distribution_logps_mask = reference_distribution_logps > reference_distribution_logps_quantile
        distribution_logps_mask = distribution_logps > distribution_logps_quantile

        reference_distribution_logps_VaR = reference_distribution_logps_quantile * reference_distribution_logps_mask.float()
        distribution_logps_VaR = distribution_logps_quantile * distribution_logps_mask.float()

        distribution = reference_distribution_logps_VaR - distribution_logps_VaR

        reference_distribution_logps_risk = reference_distribution_logps_VaR
        distribution_logps_risk = distribution_logps_VaR

        weighted_losses_above_VaR = probabilities * distribution
        CVaR = weighted_losses_above_VaR

    return CVaR, reference_distribution_logps_risk, distribution_logps_risk


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2
    else:
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False, token_level: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if token_level:
        weights = weights[:, 1:].clone()
        batch_logps = (per_token_logps * loss_mask * weights).sum(-1)
    else:
        batch_logps = (per_token_logps * loss_mask).sum(-1)

    if average_log_prob:
        return batch_logps / loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_tisdpo(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits (for TI-SDPO)."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()

    if average_log_prob:
        return (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * weights * loss_mask).sum(-1), \
            (per_position_kl * weights * loss_mask).sum(-1), \
            (per_token_logps * weights * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor."""
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, transform_config=None):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
            seed=seed,
            reverse_dataset=config.reverse_dataset,
            base_data_dir=config.base_data_dir,
        )

        self.policy = policy
        self.reference_model = reference_model
        self.transform_config = transform_config

        print(self.transform_config)

        from preference_datasets import get_dataset
        total_examples = 0
        for name in config.datasets:
            dataset = get_dataset(name, 'train', tokenizer=self.tokenizer, silent=True, cache_dir=None,
                                transform_config=transform_config,
                                base_data_dir=config.base_data_dir,
                                reverse_dataset=config.reverse_dataset)
            for prompt, data in dataset.items():
                total_examples += len(data.get('pairs', [1]))

        self.examples_per_epoch = total_examples
        rank0_print(f'Total examples per epoch: {self.examples_per_epoch}')

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, transform_config=transform_config)
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, transform_config=transform_config)
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    # =========================================================================
    # Triplet loss helpers
    # =========================================================================

    def _left_pad_for_generation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a batch so decoder-only generation starts from real tokens."""
        pad_token_id = int(self.tokenizer.pad_token_id)
        batch_size, seq_len = input_ids.shape
        out_ids = torch.full_like(input_ids, pad_token_id)
        out_mask = torch.zeros_like(attention_mask)
        lengths = attention_mask.to(torch.long).sum(dim=1)
        for i in range(batch_size):
            length_i = int(lengths[i].item())
            if length_i <= 0:
                continue
            tokens = input_ids[i][attention_mask[i].bool()]
            if tokens.numel() == 0:
                continue
            length_i = int(tokens.numel())
            out_ids[i, seq_len - length_i:] = tokens
            out_mask[i, seq_len - length_i:] = 1
        return out_ids, out_mask

    def _generate_anchor_outputs(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.LongTensor:
        """Sample an anchor response from the current policy for use in triplet loss."""
        try:
            anchor_top_k       = int(getattr(self.config.loss, 'anchor_top_k', 50))
            anchor_top_p       = float(getattr(self.config.loss, 'anchor_top_p', 0.95))
            anchor_temperature = float(getattr(self.config.loss, 'anchor_temperature', 0.8))
            max_new_tokens_cap = int(getattr(self.config.loss, 'anchor_max_new_tokens', 64))

            if 'prompt_input_ids' in batch and 'prompt_attention_mask' in batch:
                input_ids, attention_mask = self._left_pad_for_generation(
                    batch['prompt_input_ids'], batch['prompt_attention_mask'])
                max_len = int(attention_mask.to(torch.long).sum(dim=1).max().item())
            else:
                input_ids, attention_mask = self._left_pad_for_generation(
                    batch['chosen_input_ids'], batch['chosen_attention_mask'])
                max_len = int(attention_mask.to(torch.long).sum(dim=1).max().item())

            max_new_tokens = min(max_new_tokens_cap, max(1, self.config.max_length - max_len))

            ctx = lambda: (FSDP.summon_full_params(model, writeback=False, recurse=False)
                           if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx(), torch.no_grad():
                anchor_outputs = model.generate(
                    input_ids, attention_mask=attention_mask,
                    do_sample=True, top_k=anchor_top_k, top_p=anchor_top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=anchor_temperature,
                )
            return pad_to_length(anchor_outputs, self.config.max_length, self.tokenizer.pad_token_id)
        except Exception as e:
            print(f"Anchor generation failed: {e}")
            return batch['chosen_input_ids']

    def _get_log_ratio_sequence(self, model: nn.Module, reference_model: nn.Module,
                                input_ids: torch.LongTensor,
                                token_mask: Optional[torch.Tensor] = None,
                                policy_requires_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute packed per-token log-ratio sequence log(pi_theta/pi_ref) over response tokens."""
        try:
            batch_size, seq_len = input_ids.shape
            if seq_len <= 1:
                z = torch.zeros(batch_size, 0, device=input_ids.device)
                return z, torch.zeros_like(z, dtype=torch.bool)

            vocab_size = model.config.vocab_size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            if token_mask is None:
                token_mask = attention_mask.to(torch.bool)

            if policy_requires_grad:
                logits = model(input_ids, attention_mask=attention_mask).logits
            else:
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                ref_logits = reference_model(input_ids, attention_mask=attention_mask).logits

            seq_ids = input_ids[..., 1:].clamp(0, vocab_size - 1)
            logp_t  = torch.gather(F.log_softmax(logits,     dim=-1)[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)
            refp_t  = torch.gather(F.log_softmax(ref_logits, dim=-1)[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)
            log_ratio = (logp_t - refp_t) * token_mask[..., 1:].to(logp_t.dtype)

            # Pack only response-token positions
            packed, max_kept = [], 0
            for i in range(batch_size):
                kept = log_ratio[i][token_mask[i, 1:]]
                packed.append(kept)
                max_kept = max(max_kept, kept.numel())

            if max_kept == 0:
                z = torch.zeros(batch_size, 0, device=input_ids.device, dtype=log_ratio.dtype)
                return z, torch.zeros_like(z, dtype=torch.bool)

            out_batch, out_mask = [], []
            for kept in packed:
                k = kept.numel()
                pad_len = max_kept - k
                out_batch.append(torch.cat([kept, kept.new_zeros(pad_len)]))
                out_mask.append(torch.cat([
                    torch.ones(k, device=kept.device, dtype=torch.bool),
                    torch.zeros(pad_len, device=kept.device, dtype=torch.bool),
                ]))
            return torch.stack(out_batch), torch.stack(out_mask)

        except Exception as e:
            print(f"Log-ratio sequence failed: {e}")
            z = torch.zeros(input_ids.shape[0], max(0, input_ids.shape[1] - 1), device=input_ids.device)
            return z, torch.zeros_like(z, dtype=torch.bool)

    def _log_ratio_sequence_from_logits(self, logits: torch.Tensor, ref_logits: torch.Tensor,
                                        input_ids: torch.LongTensor,
                                        token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as _get_log_ratio_sequence but reuses already-computed logits (avoids extra forward passes)."""
        batch_size, seq_len = input_ids.shape
        if seq_len <= 1:
            z = torch.zeros(batch_size, 0, device=input_ids.device)
            return z, torch.zeros_like(z, dtype=torch.bool)

        vocab_size = logits.shape[-1]
        seq_ids = input_ids[..., 1:].clamp(0, vocab_size - 1)
        logp_t  = torch.gather(F.log_softmax(logits,     dim=-1)[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)
        refp_t  = torch.gather(F.log_softmax(ref_logits, dim=-1)[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)
        mask = token_mask.to(torch.bool)[..., 1:]
        log_ratio = (logp_t - refp_t) * mask.to(logp_t.dtype)

        packed, max_kept = [], 0
        for i in range(batch_size):
            kept = log_ratio[i][mask[i]]
            packed.append(kept)
            max_kept = max(max_kept, kept.numel())

        if max_kept == 0:
            z = torch.zeros(batch_size, 0, device=input_ids.device, dtype=log_ratio.dtype)
            return z, torch.zeros_like(z, dtype=torch.bool)

        out_batch, out_mask = [], []
        for kept in packed:
            k = kept.numel()
            pad_len = max_kept - k
            out_batch.append(torch.cat([kept, kept.new_zeros(pad_len)]))
            out_mask.append(torch.cat([
                torch.ones(k, device=kept.device, dtype=torch.bool),
                torch.zeros(pad_len, device=kept.device, dtype=torch.bool),
            ]))
        return torch.stack(out_batch), torch.stack(out_mask)
    '''
    def _compute_triplet_loss(self, model: nn.Module, reference_model: nn.Module,
                              batch: Dict[str, Union[List, torch.LongTensor]],
                              chosen_logits: Optional[torch.Tensor] = None,
                              rejected_logits: Optional[torch.Tensor] = None,
                              chosen_ref_logits: Optional[torch.Tensor] = None,
                              rejected_ref_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Triplet loss in log-ratio space:
            L = ReLU(||d_anchor - d_chosen||^2 - ||d_anchor - d_rejected||^2 + alpha_triplet)

        Anchor is a fresh sample from the policy; chosen = positive; rejected = negative.
        d_x = per-token log(pi_theta / pi_ref) over response tokens.

        chosen_logits / rejected_logits / chosen_ref_logits / rejected_ref_logits are
        optional pre-computed logits from the main forward pass — if provided they are
        reused to avoid extra forward passes for chosen/rejected.
        """
        try:
            pad_id = int(self.tokenizer.pad_token_id)

            # ── Anchor ────────────────────────────────────────────────────────
            anchor_ids = self._generate_anchor_outputs(model, batch)
            if 'prompt_input_ids' in batch:
                prompt_len  = int(batch['prompt_input_ids'].shape[1])
                anchor_mask = torch.zeros_like(anchor_ids, dtype=torch.bool)
                if anchor_ids.shape[1] > prompt_len:
                    anchor_mask[:, prompt_len:] = (anchor_ids[:, prompt_len:] != pad_id)
            else:
                anchor_mask = (anchor_ids != pad_id)

            d_anchor, m_anchor = self._get_log_ratio_sequence(
                model, reference_model, anchor_ids,
                token_mask=anchor_mask, policy_requires_grad=True,
            )

            # ── Positive (chosen) ─────────────────────────────────────────────
            pos_mask = (batch['chosen_labels'] != -100) if 'chosen_labels' in batch else (batch['chosen_input_ids'] != pad_id)
            if chosen_logits is not None and chosen_ref_logits is not None:
                d_pos, m_pos = self._log_ratio_sequence_from_logits(
                    chosen_logits, chosen_ref_logits, batch['chosen_input_ids'], pos_mask)
            else:
                d_pos, m_pos = self._get_log_ratio_sequence(
                    model, reference_model, batch['chosen_input_ids'],
                    token_mask=pos_mask, policy_requires_grad=True,
                )

            # ── Negative (rejected) ───────────────────────────────────────────
            neg_mask = (batch['rejected_labels'] != -100) if 'rejected_labels' in batch else (batch['rejected_input_ids'] != pad_id)
            if rejected_logits is not None and rejected_ref_logits is not None:
                d_neg, m_neg = self._log_ratio_sequence_from_logits(
                    rejected_logits, rejected_ref_logits, batch['rejected_input_ids'], neg_mask)
            else:
                d_neg, m_neg = self._get_log_ratio_sequence(
                    model, reference_model, batch['rejected_input_ids'],
                    token_mask=neg_mask, policy_requires_grad=True,
                )

            if d_anchor.numel() == 0 or d_pos.numel() == 0 or d_neg.numel() == 0:
                return d_anchor.new_tensor(0.0)

            # ── Pad all three to the same length ──────────────────────────────
            max_len = max(d_anchor.shape[1], d_pos.shape[1], d_neg.shape[1])

            def _pad(x, m, L):
                if x.shape[1] >= L:
                    return x, m
                p  = x.new_zeros(x.shape[0], L - x.shape[1])
                pm = torch.zeros(x.shape[0], L - x.shape[1], device=x.device, dtype=torch.bool)
                return torch.cat([x, p], 1), torch.cat([m, pm], 1)

            d_anchor, m_anchor = _pad(d_anchor, m_anchor, max_len)
            d_pos,    m_pos    = _pad(d_pos,    m_pos,    max_len)
            d_neg,    m_neg    = _pad(d_neg,    m_neg,    max_len)

            if torch.isnan(d_anchor).any() or torch.isnan(d_pos).any() or torch.isnan(d_neg).any():
                print("Warning: NaN in log-ratio sequences, skipping triplet loss")
                return d_anchor.new_tensor(0.0)

            # ── Hinge loss ────────────────────────────────────────────────────
            mask_pos = (m_anchor & m_pos).to(d_pos.dtype)
            mask_neg = (m_anchor & m_neg).to(d_neg.dtype)
            dist_pos = ((d_anchor - d_pos) ** 2 * mask_pos).sum(-1)
            dist_neg = ((d_anchor - d_neg) ** 2 * mask_neg).sum(-1)

            alpha_triplet = float(getattr(self.config.loss, 'alpha_triplet', 0.1))
            triplet_loss  = F.relu(dist_pos - dist_neg + alpha_triplet).mean()

            if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                print("Warning: triplet loss is NaN/Inf, returning zero")
                return d_anchor.new_tensor(0.0)

            return triplet_loss

        except Exception as e:
            print(f"Triplet loss calculation failed: {e}")
            return torch.tensor(0.0, device=next(model.parameters()).device)
    '''
    def _pack_weights(
        self,
        weights: Optional[torch.Tensor],
        token_mask: torch.Tensor,
        packed_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pack per-token weights to align with packed log-ratio sequences.

        Log-ratio packing uses ``token_mask[..., 1:]`` (shifted by 1) to select
        response positions. We apply the same shift here so that
        ``packed_weights[i, k]`` is the weight for the k-th valid response token
        of example i, matching ``d_pos[i, k]`` / ``d_neg[i, k]``.

        Args:
            weights:    [B, seq_len] weight tensor from the batch.
                        If None, uniform weights of 1.0 are returned.
            token_mask: [B, seq_len] boolean mask (before the 1-shift) that was
                        used to pack the corresponding log-ratio sequence.
            packed_len: target packed length (== d_pos.shape[1] or d_neg.shape[1]).
            device:     target device.

        Returns:
            Tensor of shape [B, packed_len].
        """
        batch_size = token_mask.shape[0]

        if weights is None:
            return torch.ones(batch_size, packed_len, device=device)

        # Shift by 1 — same convention used in _get_batch_logps / log-ratio packing
        w    = weights[:, 1:].to(device=device, dtype=torch.float32)
        mask = token_mask[:, 1:].to(torch.bool)

        out = []
        for i in range(batch_size):
            kept = w[i][mask[i]]          # valid response-token weights
            k    = kept.numel()
            if k == 0:
                out.append(torch.ones(packed_len, device=device))
            elif k >= packed_len:
                out.append(kept[:packed_len])
            else:
                # Pad with 1.0 (neutral weight) to reach packed_len
                out.append(torch.cat([kept, kept.new_ones(packed_len - k)]))

        return torch.stack(out)


    def _compute_triplet_loss(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        chosen_logits: Optional[torch.Tensor] = None,
        rejected_logits: Optional[torch.Tensor] = None,
        chosen_ref_logits: Optional[torch.Tensor] = None,
        rejected_ref_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted triplet loss in log-ratio space.

        Following the formula:

            L = E[ max(0,
                sum_t  w^w_t * (lr_anchor_t - lr_chosen_t)     [over chosen positions]
                - sum_t  w^l_t * (lr_rejected_t - lr_anchor_t)   [over rejected positions]
                + alpha_triplet
            ) ]

        where lr_x_t = log(pi_theta(y_t|x,y<t) / pi_ref(y_t|x,y<t)).

        Per-token weights w^w_t / w^l_t are taken from
        ``batch['chosen_weight']`` / ``batch['rejected_weight']`` respectively
        (falls back to uniform 1.0 when absent).

        chosen_logits / rejected_logits / chosen_ref_logits / rejected_ref_logits
        are optional pre-computed logits from the main forward pass — reused to
        avoid redundant forward passes.
        """
        try:
            pad_id = int(self.tokenizer.pad_token_id)

            # ── Anchor ────────────────────────────────────────────────────────
            anchor_ids = self._generate_anchor_outputs(model, batch)
            if 'prompt_input_ids' in batch:
                prompt_len  = int(batch['prompt_input_ids'].shape[1])
                anchor_mask = torch.zeros_like(anchor_ids, dtype=torch.bool)
                if anchor_ids.shape[1] > prompt_len:
                    anchor_mask[:, prompt_len:] = (anchor_ids[:, prompt_len:] != pad_id)
            else:
                anchor_mask = (anchor_ids != pad_id)

            d_anchor, m_anchor = self._get_log_ratio_sequence(
                model, reference_model, anchor_ids,
                token_mask=anchor_mask, policy_requires_grad=True,
            )

            # ── Positive / chosen ─────────────────────────────────────────────
            pos_mask = (
                (batch['chosen_labels'] != -100)
                if 'chosen_labels' in batch
                else (batch['chosen_input_ids'] != pad_id)
            )
            if chosen_logits is not None and chosen_ref_logits is not None:
                d_pos, m_pos = self._log_ratio_sequence_from_logits(
                    chosen_logits, chosen_ref_logits,
                    batch['chosen_input_ids'], pos_mask,
                )
            else:
                d_pos, m_pos = self._get_log_ratio_sequence(
                    model, reference_model, batch['chosen_input_ids'],
                    token_mask=pos_mask, policy_requires_grad=True,
                )

            # ── Negative / rejected ───────────────────────────────────────────
            neg_mask = (
                (batch['rejected_labels'] != -100)
                if 'rejected_labels' in batch
                else (batch['rejected_input_ids'] != pad_id)
            )
            if rejected_logits is not None and rejected_ref_logits is not None:
                d_neg, m_neg = self._log_ratio_sequence_from_logits(
                    rejected_logits, rejected_ref_logits,
                    batch['rejected_input_ids'], neg_mask,
                )
            else:
                d_neg, m_neg = self._get_log_ratio_sequence(
                    model, reference_model, batch['rejected_input_ids'],
                    token_mask=neg_mask, policy_requires_grad=True,
                )

            if d_anchor.numel() == 0 or d_pos.numel() == 0 or d_neg.numel() == 0:
                return d_anchor.new_tensor(0.0)

            # ── Pack per-token weights from batch ─────────────────────────────
            device = d_pos.device
            w_pos = self._pack_weights(
                batch.get('chosen_weight'),   pos_mask, d_pos.shape[1], device)
            w_neg = self._pack_weights(
                batch.get('rejected_weight'), neg_mask, d_neg.shape[1], device)

            # ── Pad all tensors to the same length ────────────────────────────
            max_len = max(d_anchor.shape[1], d_pos.shape[1], d_neg.shape[1])

            def _pad(x, m, L):
                if x.shape[1] >= L:
                    return x, m
                p  = x.new_zeros(x.shape[0], L - x.shape[1])
                pm = torch.zeros(x.shape[0], L - x.shape[1], device=x.device, dtype=torch.bool)
                return torch.cat([x, p], 1), torch.cat([m, pm], 1)

            def _pad_w(w, L):
                if w.shape[1] >= L:
                    return w
                return torch.cat([w, w.new_ones(w.shape[0], L - w.shape[1])], 1)

            d_anchor, m_anchor = _pad(d_anchor, m_anchor, max_len)
            d_pos,    m_pos    = _pad(d_pos,    m_pos,    max_len)
            d_neg,    m_neg    = _pad(d_neg,    m_neg,    max_len)
            w_pos              = _pad_w(w_pos, max_len)
            w_neg              = _pad_w(w_neg, max_len)

            if (torch.isnan(d_anchor).any()
                    or torch.isnan(d_pos).any()
                    or torch.isnan(d_neg).any()):
                print("Warning: NaN in log-ratio sequences, skipping triplet loss")
                return d_anchor.new_tensor(0.0)

            # ── Weighted triplet loss ─────────────────────────────────────────
            # Valid positions: anchor must also have a real token at that slot
            mask_pos = (m_anchor & m_pos).to(d_pos.dtype)   # [B, max_len]
            mask_neg = (m_anchor & m_neg).to(d_neg.dtype)   # [B, max_len]

            # term1 = sum_t  w^w_t * (lr_anchor_t - lr_chosen_t)
            term1 = (w_pos * (d_anchor - d_pos).abs() * mask_pos).sum(-1)   # [B]

            # term2 = sum_t  w^l_t * (lr_rejected_t - lr_anchor_t)
            term2 = (w_neg * (d_neg - d_anchor).abs() * mask_neg).sum(-1)   # [B]

            alpha_triplet = float(getattr(self.config.loss, 'alpha_triplet', 0.1))
            triplet_loss  = F.relu(term1 - term2 + alpha_triplet).mean()

            if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                print("Warning: triplet loss is NaN/Inf, returning zero")
                return d_anchor.new_tensor(0.0)

            return triplet_loss

        except Exception as e:
            print(f"Triplet loss calculation failed: {e}")
            return torch.tensor(0.0, device=next(model.parameters()).device)
    # =========================================================================

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], concatenated_batch['concatenated_weight'], average_log_prob=False, token_level=self.config.loss.token_level)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def tisdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps_margin, all_position_kl, all_logps = _get_batch_logps_tisdpo(all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], concatenated_batch['concatenated_weight'], average_log_prob=False)
        bsz = batch['chosen_input_ids'].shape[0]
        return all_logps_margin[:bsz], all_logps_margin[bsz:], all_position_kl[:bsz], all_position_kl[bsz:], all_logps[:bsz].detach(), all_logps[bsz:].detach()

    def tdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps_margin, all_position_kl, all_logps = _tdpo_get_batch_logps(all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        bsz = batch['chosen_input_ids'].shape[0]
        return all_logps_margin[:bsz], all_logps_margin[bsz:], all_position_kl[:bsz], all_position_kl[bsz:], all_logps[:bsz].detach(), all_logps[bsz:].detach()

    def radpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                   batch: Dict[str, Union[List, torch.LongTensor]]):
        """Forward pass for Ra-DPO, optionally computing the triplet loss.

        Returns:
            chosen_logps_margin, rejected_logps_margin,
            chosen_position_kl, rejected_position_kl,
            chosen_position_risk_ratio, rejected_position_risk_ratio,
            chosen_logps, rejected_logps,
            triplet_loss  (None when config.loss.use_triplet is False/absent)
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)

        all_logps_margin, all_position_kl, all_position_risk_ratio, all_logps = _radpo_get_batch_logps(
            all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], concatenated_batch['concatenated_weight'],
            confidence_level=self.config.loss.confidence_level,
            is_split_risk_ratio=self.config.loss.is_split_risk_ratio,
            is_cal_risk_distribution_logps=self.config.loss.is_cal_risk_distribution_logps,
            average_log_prob=False,
        )

        bsz = batch['chosen_input_ids'].shape[0]
        chosen_logps_margin          = all_logps_margin[:bsz]
        rejected_logps_margin        = all_logps_margin[bsz:]
        chosen_position_kl           = all_position_kl[:bsz]
        rejected_position_kl         = all_position_kl[bsz:]
        chosen_position_risk_ratio   = all_position_risk_ratio[:bsz]
        rejected_position_risk_ratio = all_position_risk_ratio[bsz:]
        chosen_logps                 = all_logps[:bsz].detach()
        rejected_logps               = all_logps[bsz:].detach()

        # ── Triplet loss (opt-in via config.loss.use_triplet) ─────────────────
        use_triplet   = getattr(self.config.loss, 'use_triplet', False)
        alpha_triplet = float(getattr(self.config.loss, 'alpha_triplet', 0.1))

        if use_triplet and alpha_triplet > 0:
            triplet_loss = self._compute_triplet_loss(
                model, reference_model, batch,
                chosen_logits=all_logits[:bsz],
                rejected_logits=all_logits[bsz:],
                chosen_ref_logits=reference_all_logits[:bsz],
                rejected_ref_logits=reference_all_logits[bsz:],
            )
        else:
            triplet_loss = None

        return (chosen_logps_margin, rejected_logps_margin,
                chosen_position_kl, rejected_position_kl,
                chosen_position_risk_ratio, rejected_position_risk_ratio,
                chosen_logps, rejected_logps,
                triplet_loss)

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards    = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards  = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen']     = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected']   = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins']    = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'tdpo':
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps \
                = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tdpo_loss(chosen_logps_margin, rejected_logps_margin,
                                                                 chosen_position_kl, rejected_position_kl,
                                                                 beta=loss_config.beta, alpha=loss_config.alpha, if_tdpo2=loss_config.if_tdpo2)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards    = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards  = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen']     = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected']   = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins']    = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl   = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)
            metrics[f'kl_{train_test}/chosen']  = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin']  = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'tisdpo':
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps \
                = self.tisdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tisdpo_loss(chosen_logps_margin, rejected_logps_margin,
                                                                   chosen_position_kl, rejected_position_kl,
                                                                   beta=loss_config.beta, alpha=loss_config.alpha, token_level=loss_config.token_level)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards    = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards  = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen']     = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected']   = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins']    = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl   = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)
            metrics[f'kl_{train_test}/chosen']  = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin']  = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'radpo':
            (chosen_logps_margin, rejected_logps_margin,
             chosen_position_kl, rejected_position_kl,
             chosen_position_risk_ratio, rejected_position_risk_ratio,
             policy_chosen_logps, policy_rejected_logps,
             triplet_loss) = self.radpo_concatenated_forward(self.policy, self.reference_model, batch)

            # Base Ra-DPO loss
            base_losses, chosen_rewards, rejected_rewards = radpo_loss(
                chosen_logps_margin, rejected_logps_margin,
                chosen_position_risk_ratio, rejected_position_risk_ratio,
                beta=loss_config.beta, alpha=loss_config.alpha, if_radpo2=loss_config.if_radpo2,
            )

            # Add triplet loss weighted by gamma
            gamma = float(getattr(loss_config, 'gamma', 0.1))
            if triplet_loss is not None:
                losses = base_losses + gamma * triplet_loss
                metrics[f'triplet_loss_{train_test}'] = [triplet_loss.detach().cpu().item()]
            else:
                losses = base_losses
                metrics[f'triplet_loss_{train_test}'] = [0.0]
            metrics[f'gamma_{train_test}'] = [gamma]

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards    = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards  = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen']     = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected']   = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins']    = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl   = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)
            metrics[f'kl_{train_test}/chosen']  = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin']  = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            all_device_chosen_risk_ratio   = all_gather_if_needed(chosen_position_risk_ratio.detach(), self.rank, self.world_size)
            all_device_rejected_risk_ratio = all_gather_if_needed(rejected_position_risk_ratio.detach(), self.rank, self.world_size)
            metrics[f'risk_ratio_{train_test}/chosen']  = all_device_chosen_risk_ratio.cpu().numpy().tolist()
            metrics[f'risk_ratio_{train_test}/rejected'] = all_device_rejected_risk_ratio.cpu().numpy().tolist()
            metrics[f'risk_ratio_{train_test}/margin']  = (all_device_chosen_risk_ratio - all_device_rejected_risk_ratio).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps  = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False, token_level=False)
            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        self.current_epoch = 0
        last_log = None
        last_saved_epoch = 0

        rank0_print(f'Starting training with {self.examples_per_epoch} examples per epoch')
        rank0_print(f'save_every_epoch: {self.config.get("save_every_epoch", True)}')

        for batch in self.train_iterator:
            if self.examples_per_epoch > 0:
                self.current_epoch = self.example_counter // self.examples_per_epoch

            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples (Epoch {self.current_epoch})')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)
                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)
                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)
                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)
                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch  = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()
                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if self.examples_per_epoch > 0:
                new_epoch = self.example_counter // self.examples_per_epoch
                save_every_epoch = self.config.get('save_every_epoch', True) if hasattr(self.config, 'get') else getattr(self.config, 'save_every_epoch', True)
                if save_every_epoch and new_epoch > last_saved_epoch:
                    self.current_epoch = new_epoch
                    rank0_print(f'\n{"="*60}\nCompleted Epoch {new_epoch}\n{"="*60}')
                    checkpoint_dir = os.path.join(self.run_dir, f'checkpoint-epoch-{new_epoch}')
                    mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                    mean_train_metrics['epoch'] = new_epoch
                    self.save(output_dir=checkpoint_dir, metrics=mean_train_metrics)
                    rank0_print(f'Checkpoint saved to {checkpoint_dir}\n{"="*60}\n')
                    last_saved_epoch = new_epoch

            save_every_steps = self.config.get('save_every_steps', 0) if hasattr(self.config, 'get') else getattr(self.config, 'save_every_steps', 0)
            if save_every_steps > 0 and self.example_counter % save_every_steps == 0 and self.example_counter > 0:
                checkpoint_dir = os.path.join(self.run_dir, f'checkpoint-step-{self.example_counter}')
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                self.save(output_dir=checkpoint_dir, metrics=mean_train_metrics)
                rank0_print(f'Checkpoint saved to {checkpoint_dir}')

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates']  = self.batch_counter
                mean_train_metrics['counters/epoch']    = self.current_epoch
                epoch_progress = (self.example_counter % self.examples_per_epoch) / self.examples_per_epoch * 100 if self.examples_per_epoch > 0 else 0
                rank0_print(f'train stats after {self.example_counter} examples (Epoch {self.current_epoch}, {epoch_progress:.1f}% complete): {formatted_dict(mean_train_metrics)}')
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)
                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

    def clip_gradient(self):
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({'step_idx': step, 'state': state, 'metrics': metrics if metrics is not None else {}}, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        model_save_dir = output_dir if output_dir is not None else os.path.join(self.run_dir, 'LATEST')
        os.makedirs(model_save_dir, exist_ok=True)
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        self.tokenizer.save_pretrained(model_save_dir)
        if metrics is not None:
            with open(os.path.join(model_save_dir, "training_metrics.json"), "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, transform_config=None):
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, transform_config=transform_config)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper, offload_to_cpu=False, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            model_save_dir = output_dir if output_dir is not None else self.run_dir
            os.makedirs(model_save_dir, exist_ok=True)
            from transformers import AutoModelForCausalLM
            unwrapped_model = AutoModelForCausalLM.from_pretrained(self.config.model.name_or_path)
            unwrapped_model.load_state_dict(policy_state_dict)
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
            del unwrapped_model
            self.tokenizer.save_pretrained(model_save_dir)
            if metrics is not None:
                with open(os.path.join(model_save_dir, "training_metrics.json"), "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)

        del policy_state_dict
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1, transform_config=None):
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, transform_config=transform_config)
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'radpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
        model_save_dir = output_dir if output_dir is not None else os.path.join(self.run_dir, 'LATEST')
        os.makedirs(model_save_dir, exist_ok=True)
        from transformers import AutoModelForCausalLM
        unwrapped_model = AutoModelForCausalLM.from_pretrained(self.config.model.name_or_path)
        unwrapped_model.load_state_dict(policy_state_dict)
        unwrapped_model.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        del unwrapped_model
        self.tokenizer.save_pretrained(model_save_dir)
        if metrics is not None:
            with open(os.path.join(model_save_dir, "training_metrics.json"), "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)
        del policy_state_dict
        