#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

IMAGE_TOKEN_INDEX = -200


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        self.early_exit_callback = None  # 用于early exit的回调函数
        self.early_exit_start_layer = 16  # 从第16层开始检查

    def _orion_prepare_causal_mask(
        self,
        attention_mask,
        hidden_states,
        cache_position,
        past_key_values,
        output_attentions,
        past_key_values_length,
    ):
        if hasattr(super(), "_update_causal_mask"):
            return super()._update_causal_mask(
                attention_mask, hidden_states, cache_position,
                past_key_values, output_attentions)
        if hasattr(self, "_prepare_decoder_attention_mask"):
            return self._prepare_decoder_attention_mask(
                attention_mask,
                (hidden_states.shape[0], hidden_states.shape[1]),
                hidden_states,
                past_key_values_length,
            )
        return attention_mask

    @staticmethod
    def _orion_layer_accepts(decoder_layer, name):
        return name in inspect.signature(decoder_layer.forward).parameters
    
    def set_early_exit_callback(self, callback):
        """设置early exit回调函数，callback(hidden_states, layer_idx) -> bool，返回True表示应该early exit"""
        self.early_exit_callback = callback

    def _autoprune_enabled(self, token_ids, past_key_values, use_cache):
        if os.environ.get("ORION_AUTOPRUNE", "0") != "1":
            return False
        if os.environ.get("ORION_VLA_PRUNER", "0") == "1":
            raise ValueError("ORION_AUTOPRUNE and ORION_VLA_PRUNER are mutually exclusive.")
        if self.training or past_key_values is not None or use_cache:
            return False
        if token_ids is None or token_ids.shape[0] != 1:
            return False
        return bool((token_ids == IMAGE_TOKEN_INDEX).any())

    def _vla_pruner_enabled(self, token_ids, past_key_values, use_cache):
        if os.environ.get("ORION_VLA_PRUNER", "0") != "1":
            return False
        if os.environ.get("ORION_AUTOPRUNE", "0") == "1":
            raise ValueError("ORION_AUTOPRUNE and ORION_VLA_PRUNER are mutually exclusive.")
        if self.training or past_key_values is not None or use_cache:
            return False
        if token_ids is None or token_ids.shape[0] != 1:
            return False
        return bool((token_ids == IMAGE_TOKEN_INDEX).any())

    def _autoprune_config(self):
        return {
            "target_token_num": int(os.environ.get("ORION_AUTOPRUNE_TARGET_TOKEN_NUM", "64")),
            "x0": float(os.environ.get("ORION_AUTOPRUNE_X0", "14.9")),
            "k0": float(os.environ.get("ORION_AUTOPRUNE_K0", "0.4")),
            "gamma": float(os.environ.get("ORION_AUTOPRUNE_GAMMA", "0.2")),
        }

    @staticmethod
    def _autoprune_mutual_information(attn_block, eps=1e-6):
        # attn_block: [B, H, text_tokens, visual_tokens], already normalized by attention softmax.
        x = attn_block.float().clamp_min(eps)
        tx = x.shape[-2]
        px = 1.0 / max(tx, 1)
        p_xy = x * px
        p_y = p_xy.sum(dim=-2, keepdim=True)
        ratio = p_xy / (px * p_y + eps)
        return (p_xy * torch.log2(ratio.clamp_min(eps))).sum(dim=(-1, -2)).sum(dim=1)

    @staticmethod
    def _autoprune_keep_schedule(pruning_loc, v_token_num, seq_len, target_token_num, dynamic_k, x0):
        def logistic(pos_idx):
            return 0.75 / (1.0 + math.exp(dynamic_k * (pos_idx - (x0 + dynamic_k))))

        keep_percent = [logistic(pos_idx) for pos_idx in pruning_loc]
        target_sum = seq_len * target_token_num

        def token_cost(scale):
            total = v_token_num
            current = v_token_num
            for idx, percent in enumerate(keep_percent):
                keep_target = min(math.ceil((v_token_num - 1) * min(1.0, percent * scale)), v_token_num)
                current = max(1, min(current, keep_target))
                prev_pos = 0 if idx == 0 else pruning_loc[idx - 1]
                total += current * max(pruning_loc[idx] - prev_pos, 1)
            total += current * max(seq_len - pruning_loc[-1], 0)
            return total

        low, high = 0.0, 20.0
        best_scale, best_diff = 1.0, float("inf")
        for _ in range(80):
            mid = (low + high) / 2
            cost = token_cost(mid)
            diff = abs(cost - target_sum)
            if diff < best_diff:
                best_scale, best_diff = mid, diff
            if cost < target_sum:
                low = mid
            else:
                high = mid
        return [min(1.0, max(0.0, percent * best_scale)) for percent in keep_percent]

    def _autoprune_step(self, hidden_states, token_ids, attention_mask, layer_attn, layer_idx, state):
        visual_mask = token_ids == IMAGE_TOKEN_INDEX
        visual_idx = torch.where(visual_mask[0])[0]
        if visual_idx.numel() <= 1:
            return hidden_states, token_ids, attention_mask, state

        text_idx = torch.where(~visual_mask[0] & attention_mask[0].bool())[0]
        text_idx = text_idx[text_idx > visual_idx[-1]]
        if text_idx.numel() == 0:
            return hidden_states, token_ids, attention_mask, state

        pruning_loc = state["pruning_loc"]
        if layer_idx not in pruning_loc:
            return hidden_states, token_ids, attention_mask, state

        if state["keep_schedule"] is None:
            attn_block = layer_attn[:, :, text_idx, :][:, :, :, visual_idx]
            mi = self._autoprune_mutual_information(attn_block).mean().item()
            cfg = state["config"]
            dynamic_k = max((-cfg["gamma"] * mi) + cfg["k0"], 0.0)
            state["keep_schedule"] = self._autoprune_keep_schedule(
                pruning_loc,
                state["original_visual_tokens"],
                len(self.layers),
                cfg["target_token_num"],
                dynamic_k,
                cfg["x0"],
            )
            state["dynamic_k"] = dynamic_k
            state["mutual_information"] = mi

        schedule_idx = min(state["schedule_idx"], len(state["keep_schedule"]) - 1)
        keep_percent = state["keep_schedule"][schedule_idx]
        state["schedule_idx"] += 1

        target_keep = max(1, min(visual_idx.numel(), math.ceil((state["original_visual_tokens"] - 1) * keep_percent)))
        if target_keep >= visual_idx.numel():
            state["visual_tokens_per_layer"].append(int(visual_idx.numel()))
            return hidden_states, token_ids, attention_mask, state

        with torch.no_grad():
            scores = layer_attn[:, :, text_idx, :][:, :, :, visual_idx].mean(dim=(1, 2))
            keep_local = torch.topk(scores, k=target_keep, dim=1).indices.sort(dim=1).values
            keep_visual_idx = visual_idx[keep_local[0]]
            keep_mask = torch.ones(token_ids.shape[1], dtype=torch.bool, device=token_ids.device)
            keep_mask[visual_idx] = False
            keep_mask[keep_visual_idx] = True
            keep_idx = torch.where(keep_mask)[0].unsqueeze(0)

        hidden_states = hidden_states.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
        token_ids = token_ids.gather(1, keep_idx)
        attention_mask = attention_mask.gather(1, keep_idx)
        state["visual_tokens_per_layer"].append(int(target_keep))
        return hidden_states, token_ids, attention_mask, state

    def _vla_pruner_config(self):
        return {
            "target_token_num": int(os.environ.get("ORION_VLA_PRUNER_TARGET_TOKEN_NUM", "64")),
            "prune_layer": int(os.environ.get("ORION_VLA_PRUNER_LAYER", "2")),
            "semantic_ratio": float(os.environ.get("ORION_VLA_PRUNER_SEMANTIC_RATIO", "0.5")),
            "temporal_alpha": float(os.environ.get("ORION_VLA_PRUNER_TEMPORAL_ALPHA", "0.7")),
            "use_temporal": os.environ.get("ORION_VLA_PRUNER_USE_TEMPORAL", "1") != "0",
        }

    @staticmethod
    def _vla_pruner_normalize_scores(scores, eps=1e-6):
        scores = scores.float()
        scores = scores - scores.amin(dim=-1, keepdim=True)
        denom = scores.amax(dim=-1, keepdim=True).clamp_min(eps)
        return scores / denom

    def _vla_pruner_action_indices(self, token_ids, text_idx):
        waypoint_token_idx = getattr(self.config, "waypoint_token_idx", None)
        action_mask = torch.zeros_like(token_ids[0], dtype=torch.bool)
        if waypoint_token_idx is not None:
            token_list = waypoint_token_idx if isinstance(waypoint_token_idx, list) else [waypoint_token_idx]
            for token_id in token_list:
                action_mask = torch.logical_or(action_mask, token_ids[0] == token_id)
        action_idx = torch.where(action_mask)[0]
        if action_idx.numel() == 0:
            action_idx = text_idx[-1:]
        return action_idx

    def _vla_pruner_scores(self, token_ids, attention_mask, layer_attn, visual_idx, state):
        text_idx = torch.where(~(token_ids[0] == IMAGE_TOKEN_INDEX) & attention_mask[0].bool())[0]
        text_idx = text_idx[text_idx > visual_idx[-1]]
        if text_idx.numel() == 0:
            text_idx = torch.where(~(token_ids[0] == IMAGE_TOKEN_INDEX) & attention_mask[0].bool())[0]
        if text_idx.numel() == 0:
            return None, None, None

        semantic = layer_attn[:, :, text_idx, :][:, :, :, visual_idx].mean(dim=(1, 2))
        action_idx = self._vla_pruner_action_indices(token_ids, text_idx)
        action = layer_attn[:, :, action_idx, :][:, :, :, visual_idx].mean(dim=(1, 2))

        cfg = state["config"]
        if cfg["use_temporal"]:
            prev = getattr(self, "_vla_pruner_temporal_scores", None)
            if prev is not None and prev.shape[-1] == action.shape[-1]:
                prev = prev.to(device=action.device, dtype=action.dtype)
                temporal = cfg["temporal_alpha"] * prev + (1.0 - cfg["temporal_alpha"]) * action.detach()
            else:
                temporal = action.detach()
            self._vla_pruner_temporal_scores = temporal.detach()
        else:
            temporal = action.detach()

        return semantic, temporal, action_idx

    def _vla_pruner_step(self, hidden_states, token_ids, attention_mask, layer_attn, layer_idx, state):
        visual_mask = token_ids == IMAGE_TOKEN_INDEX
        visual_idx = torch.where(visual_mask[0])[0]
        if visual_idx.numel() <= 1:
            state["visual_tokens_per_layer"].append(int(visual_idx.numel()))
            return hidden_states, token_ids, attention_mask, state

        current_layer = layer_idx + 1
        if state["pruned"] or current_layer < state["config"]["prune_layer"]:
            state["visual_tokens_per_layer"].append(int(visual_idx.numel()))
            return hidden_states, token_ids, attention_mask, state

        target_keep = max(1, min(int(state["config"]["target_token_num"]), int(visual_idx.numel())))
        if target_keep >= visual_idx.numel():
            state["pruned"] = True
            state["prune_layer"] = current_layer
            state["visual_tokens_per_layer"].append(int(visual_idx.numel()))
            return hidden_states, token_ids, attention_mask, state

        scores = self._vla_pruner_scores(token_ids, attention_mask, layer_attn, visual_idx, state)
        if scores[0] is None:
            state["visual_tokens_per_layer"].append(int(visual_idx.numel()))
            return hidden_states, token_ids, attention_mask, state

        semantic_scores, temporal_scores, action_idx = scores
        semantic_scores = self._vla_pruner_normalize_scores(semantic_scores)
        temporal_scores = self._vla_pruner_normalize_scores(temporal_scores)
        combined_scores = torch.maximum(semantic_scores, temporal_scores)

        semantic_keep = max(1, min(target_keep, int(math.ceil(target_keep * state["config"]["semantic_ratio"]))))
        temporal_keep = max(1, min(target_keep, target_keep - semantic_keep))
        with torch.no_grad():
            semantic_top = torch.topk(semantic_scores, k=semantic_keep, dim=1).indices
            temporal_top = torch.topk(temporal_scores, k=temporal_keep, dim=1).indices
            candidate_mask = torch.zeros_like(combined_scores, dtype=torch.bool)
            candidate_mask.scatter_(1, semantic_top, True)
            candidate_mask.scatter_(1, temporal_top, True)

            candidate_count = int(candidate_mask[0].sum().item())
            if candidate_count >= target_keep:
                masked_scores = combined_scores.masked_fill(~candidate_mask, float("-inf"))
                keep_local = torch.topk(masked_scores, k=target_keep, dim=1).indices.sort(dim=1).values
            else:
                keep_local = torch.topk(combined_scores, k=target_keep, dim=1).indices.sort(dim=1).values

            keep_visual_idx = visual_idx[keep_local[0]]
            keep_mask = torch.ones(token_ids.shape[1], dtype=torch.bool, device=token_ids.device)
            keep_mask[visual_idx] = False
            keep_mask[keep_visual_idx] = True
            keep_idx = torch.where(keep_mask)[0].unsqueeze(0)

        hidden_states = hidden_states.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
        token_ids = token_ids.gather(1, keep_idx)
        attention_mask = attention_mask.gather(1, keep_idx)
        state["pruned"] = True
        state["prune_layer"] = current_layer
        state["candidate_tokens"] = candidate_count
        state["action_tokens"] = int(action_idx.numel())
        state["visual_tokens_per_layer"].append(int(target_keep))
        return hidden_states, token_ids, attention_mask, state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        auto_prune_input_ids: Optional[torch.LongTensor] = None,
    ):
        """重写forward方法，支持early exit"""
        from transformers.modeling_outputs import BaseModelOutputWithPast
        
        requested_output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        autopruning = self._autoprune_enabled(auto_prune_input_ids, past_key_values, use_cache)
        vla_pruning = self._vla_pruner_enabled(auto_prune_input_ids, past_key_values, use_cache)
        token_pruning = autopruning or vla_pruning
        output_attentions = requested_output_attentions or token_pruning
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        causal_mask = self._orion_prepare_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values,
            output_attentions, past_key_values_length
        )

        hidden_states = inputs_embeds
        position_embeddings = (
            self.rotary_emb(hidden_states, position_ids)
            if hasattr(self, "rotary_emb") else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if autopruning:
            auto_prune_input_ids = auto_prune_input_ids.to(device=inputs_embeds.device)
            self._last_autoprune_input_ids = auto_prune_input_ids
            visual_tokens = int((auto_prune_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            autopruning_state = {
                "config": self._autoprune_config(),
                "pruning_loc": list(range(1, len(self.layers))),
                "original_visual_tokens": visual_tokens,
                "keep_schedule": None,
                "schedule_idx": 0,
                "dynamic_k": None,
                "mutual_information": None,
                "visual_tokens_per_layer": [visual_tokens],
            }
        else:
            self._last_autoprune_input_ids = auto_prune_input_ids
            autopruning_state = None
        if vla_pruning:
            auto_prune_input_ids = auto_prune_input_ids.to(device=inputs_embeds.device)
            self._last_autoprune_input_ids = auto_prune_input_ids
            visual_tokens = int((auto_prune_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            vla_pruner_state = {
                "config": self._vla_pruner_config(),
                "original_visual_tokens": visual_tokens,
                "visual_tokens_per_layer": [visual_tokens],
                "pruned": False,
                "prune_layer": None,
                "candidate_tokens": None,
                "action_tokens": None,
            }
        else:
            vla_pruner_state = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if requested_output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 逐层forward，支持early exit
        early_exit_triggered = False
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                if self._orion_layer_accepts(decoder_layer, "cache_position"):
                    layer_kwargs["cache_position"] = cache_position
                if (
                    position_embeddings is not None
                    and self._orion_layer_accepts(decoder_layer, "position_embeddings")
                ):
                    layer_kwargs["position_embeddings"] = position_embeddings
                layer_outputs = decoder_layer(**layer_kwargs)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if requested_output_attentions:
                all_self_attns += (layer_outputs[1],)

            if autopruning and idx < len(self.layers) - 1:
                hidden_states, auto_prune_input_ids, attention_mask, autopruning_state = self._autoprune_step(
                    hidden_states,
                    auto_prune_input_ids,
                    attention_mask,
                    layer_outputs[1],
                    idx,
                    autopruning_state,
                )
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                ).unsqueeze(0).view(-1, seq_length)
                cache_position = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                causal_mask = self._orion_prepare_causal_mask(
                    attention_mask, hidden_states, cache_position, past_key_values,
                    output_attentions, past_key_values_length
                )
                position_embeddings = (
                    self.rotary_emb(hidden_states, position_ids)
                    if hasattr(self, "rotary_emb") else None
                )
                self._last_autoprune_input_ids = auto_prune_input_ids
            elif vla_pruning and idx < len(self.layers) - 1:
                hidden_states, auto_prune_input_ids, attention_mask, vla_pruner_state = self._vla_pruner_step(
                    hidden_states,
                    auto_prune_input_ids,
                    attention_mask,
                    layer_outputs[1],
                    idx,
                    vla_pruner_state,
                )
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                ).unsqueeze(0).view(-1, seq_length)
                cache_position = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                causal_mask = self._orion_prepare_causal_mask(
                    attention_mask, hidden_states, cache_position, past_key_values,
                    output_attentions, past_key_values_length
                )
                position_embeddings = (
                    self.rotary_emb(hidden_states, position_ids)
                    if hasattr(self, "rotary_emb") else None
                )
                self._last_autoprune_input_ids = auto_prune_input_ids

            # Early exit检查：从第16层开始，每层后检查
            if self.early_exit_callback is not None and idx >= self.early_exit_start_layer - 1:
                if self.early_exit_callback(hidden_states, idx):
                    early_exit_triggered = True
                    # 如果output_hidden_states，需要补齐剩余的None
                    if output_hidden_states:
                        remaining_layers = len(self.layers) - idx - 1
                        all_hidden_states += (None,) * remaining_layers
                    break

        hidden_states = self.norm(hidden_states)
        if autopruning:
            if not hasattr(self, "_autoprune_stats"):
                self._autoprune_stats = []
            final_visual_tokens = int((auto_prune_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            self._autoprune_stats.append({
                "initial_visual_tokens": autopruning_state["original_visual_tokens"],
                "final_visual_tokens": final_visual_tokens,
                "avg_visual_tokens": float(sum(autopruning_state["visual_tokens_per_layer"]) / len(autopruning_state["visual_tokens_per_layer"])),
                "dynamic_k": autopruning_state["dynamic_k"],
                "mutual_information": autopruning_state["mutual_information"],
            })
        if vla_pruning:
            if not hasattr(self, "_vla_pruner_stats"):
                self._vla_pruner_stats = []
            final_visual_tokens = int((auto_prune_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            self._vla_pruner_stats.append({
                "initial_visual_tokens": vla_pruner_state["original_visual_tokens"],
                "final_visual_tokens": final_visual_tokens,
                "avg_visual_tokens": float(sum(vla_pruner_state["visual_tokens_per_layer"]) / len(vla_pruner_state["visual_tokens_per_layer"])),
                "prune_layer": vla_pruner_state["prune_layer"],
                "candidate_tokens": vla_pruner_state["candidate_tokens"],
                "action_tokens": vla_pruner_state["action_tokens"],
                "target_token_num": vla_pruner_state["config"]["target_token_num"],
                "temporal_alpha": vla_pruner_state["config"]["temporal_alpha"],
            })

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, use_gen_token=False, use_critical_qa=False):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

        number_tokens = [
                718,
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # +-0.123456789
        if use_gen_token:
            weighted_mask = torch.ones(self.config.vocab_size + 1)
            weighted_mask[number_tokens] = 1.0
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
        if use_critical_qa:
            weighted_mask[number_tokens] = 3.0
        self.register_buffer("weighted_mask", weighted_mask)
        self.use_gen_token = use_gen_token
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_ego_feature: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        else:
            new_input_ids = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids, # None
            attention_mask=attention_mask, # (1, 588)
            position_ids=position_ids, # None
            past_key_values=past_key_values, # None
            inputs_embeds=inputs_embeds, # (1, 588, 4096)
            use_cache=use_cache, # False
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # False
            return_dict=return_dict, # True
            auto_prune_input_ids=new_input_ids,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]
        new_input_ids = getattr(self.model, "_last_autoprune_input_ids", new_input_ids)

        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list,dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states) # (1, 588, 32001)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # 使用特殊token：如果不加特殊token不参与训练，我加了特殊token，参与训链了，然后这个类型不太对，应该是torch.float32
        # self.model.embed_tokens = self.model.embed_tokens.to(torch.float32)
        # self.model.to(torch.float32)
        if return_ego_feature:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
       

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def inference_ego(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_ego_feature = False,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            new_input_ids = inputs
        
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        # 运行模型（如果设置了early exit回调，会在forward过程中逐层检查并可能提前退出）
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=kwargs.get('use_cache', False),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            auto_prune_input_ids=new_input_ids,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]
        new_input_ids = getattr(self.model, "_last_autoprune_input_ids", new_input_ids)

        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list,dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            return selected_hidden_states
        else:
            assert False
        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

def add_special_token(special_token_list, tokenizer, model):
    # 给新的token添加索引并用大模型的embeding的平均值来初始化token的embeding
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens = True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
