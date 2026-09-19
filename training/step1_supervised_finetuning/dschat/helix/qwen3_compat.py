"""Controlled Qwen3-Dense backport for the custom Transformers 4.46 build."""

from __future__ import annotations

import torch
from torch import nn

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2 import modeling_qwen2


class Qwen3CompatConfig(Qwen2Config):
    model_type = "qwen3"

    def __init__(self, head_dim=128, attention_bias=False, **kwargs):
        self.head_dim = int(head_dim)
        self.attention_bias = bool(attention_bias)
        self.helix_qwen3_compat = True
        super().__init__(**kwargs)


class HelixQwenSdpaAttention(modeling_qwen2.Qwen2SdpaAttention):
    """Qwen2 cache/mask integration with Qwen3 projection and QK-Norm semantics."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if not getattr(config, "helix_qwen3_compat", False):
            return
        self.head_dim = int(config.head_dim)
        self.retained_query_heads = int(self.num_heads * self.head_prune_rate)
        self.retained_kv_heads = int(
            self.num_key_value_heads * self.head_prune_rate
        )
        q_width = self.retained_query_heads * self.head_dim
        kv_width = self.retained_kv_heads * self.head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            q_width,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            kv_width,
            bias=bool(config.attention_bias),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            kv_width,
            bias=bool(config.attention_bias),
        )
        self.o_proj = nn.Linear(q_width, self.hidden_size, bias=False)
        self.q_norm = modeling_qwen2.Qwen2RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = modeling_qwen2.Qwen2RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        if not getattr(self.config, "helix_qwen3_compat", False):
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        if output_attentions:
            raise ValueError(
                "Helix Qwen3 backport supports the audited SDPA path only; "
                "output_attentions=True would require an unpruned eager fallback"
            )

        batch, sequence, _ = hidden_states.shape
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(
                batch,
                sequence,
                self.retained_query_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                batch,
                sequence,
                self.retained_kv_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch,
            sequence,
            self.retained_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = modeling_qwen2.repeat_kv(
            key_states,
            self.num_key_value_groups,
        )
        value_states = modeling_qwen2.repeat_kv(
            value_states,
            self.num_key_value_groups,
        )
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and sequence > 1,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch,
            sequence,
            self.retained_query_heads * self.head_dim,
        )
        return self.o_proj(attention_output), None, past_key_value


def install_qwen3_compat_attention() -> None:
    current = modeling_qwen2.QWEN2_ATTENTION_CLASSES.get("sdpa")
    if current is HelixQwenSdpaAttention:
        return
    modeling_qwen2.QWEN2_ATTENTION_CLASSES["sdpa"] = HelixQwenSdpaAttention
