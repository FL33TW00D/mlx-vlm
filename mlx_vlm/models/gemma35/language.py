import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache, RotatingKVCache


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262208
    num_key_value_heads: int = 4
    laurel_rank: int = 64
    frac_shared_layers: float = 0.5
    altup_active_idx: int = 0
    altup_num_inputs: int = 4
    altup_coef_clip: Optional[float] = None
    altup_correct_scale: bool = True
    hidden_size_per_layer_input: int = 1024
    rope_local_base_freq: float = 10000.0
    rope_traditional: bool = False
    rope_theta: float = 1000000.0
    query_pre_attn_scalar: float = 0.0625
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 5
    activation_sparsity_pattern: Optional[List[float]] = None
    final_logit_softcapping: float = 1.0
    query_rescale_scalar: float = 1.0
    num_kv_shared_layers: int = 0

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Gemma3p5EinsumLayer(nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        einsum_str: str,
        *args,
        **kwargs,
    ):
        if "->" not in einsum_str:
            raise ValueError("Einsum must contain '->'")

        if len(einsum_str.split("->")[0].split(",")) != 2:
            raise ValueError("Need to have exactly two inputs in einsum instruction")

        super().__init__(*args, **kwargs)
        self.shape = shape
        self.einsum_str = einsum_str

        self.weight = mx.ones(shape)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return mx.einsum(self.einsum_str, x, self.weight)




class Gemma3p5RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = mx.ones(dim)
        else:
            self.weight = None

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = self._guard_against_excess_precision(x)
        if not self.with_scale:
            self.weight = mx.array(1.0)

        output = self._norm(x) * (self.weight + self.scale_shift).astype(x.dtype)
        return output.astype(x.dtype)


    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _guard_against_excess_precision(self, x: mx.array) -> mx.array:
        # TODO(ryanmullins): Implement Torch equivalent to jax.lax.reduce_precision
        return x


class Gemma3p5LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: TextConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        x = x + normed_laurel_x
        return x


class Gemma3p5Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern == 0)

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = config.query_rescale_scalar / config.query_pre_attn_scalar

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.qkv_norm = Gemma3p5RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx

        # Compute the layer index from which shared KV cache values will be retrieved.
        if not self.is_kv_shared_layer:
            self.kv_shared_layer_index = None
        elif self.is_sliding:
            # The last layer that computes local sliding attention is always 2 before sharing starts
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 2
        else:
            # The last layer before sharing starts is always the last that computes global attention layer
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 1
        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta if self.is_kv_shared_layer else config.rope_local_base_freq
        )


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(x)
        queries = queries.reshape(hidden_shape)
        queries = self.qkv_norm(queries)
        if cache is not None:
            queries = self.rope(queries, cache.offset)
        else:
            queries = self.rope(queries)
        queries = queries.transpose(0, 2, 1, 3)

        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None and cache is not None and cache.offset > 0:
            keys, values = cache.state

        else:
            keys = self.k_proj(x).reshape(hidden_shape)
            keys = self.qkv_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            if cache is not None:
                keys = self.rope(keys, cache.offset)
            else:
                keys = self.rope(keys)

            values = self.v_proj(x).reshape(hidden_shape)
            values = self.qkv_norm(values)
            values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)


            # # Sliding window
            # if self.is_sliding and mask is not None and isinstance(mask, mx.array):

            #     if mask.shape[-1] != keys.shape[-2]:
            #         mask = mask[..., -keys.shape[-2] :]

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(input_shape + (-1,))
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0, *args, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()
        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0

    def __call__(self, x: mx.array):
        gate_proj = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: mx.array) -> mx.array:
        # Calculate the cutoff value based on the target sparsity
        # For normal distribution, we use the inverse CDF (quantile function)
        # Convert to numpy, calculate the quantile, then back to mx.array
        # Use numpy's special functions instead of scipy
        if self.activation_sparsity <= 0.0:
            # For 0 sparsity, return infinity to match PyTorch behavior
            # This will make all values pass through
            inf_value = mx.array(float("inf"))
            return mx.broadcast_to(inf_value, inputs.shape)

        normal_dist = mx.random.normal((1,))

        # Generate a large sample from normal distribution
        sample_size = 100000
        normal_samples = mx.random.normal(shape=(sample_size,))

        # Sort the samples
        sorted_samples = mx.sort(normal_samples)

        # Find the index corresponding to our target sparsity
        idx = int(self.activation_sparsity * sample_size)

        # Get the value at that index as our std_multiplier
        std_multiplier = float(sorted_samples[idx]) if idx < sample_size else 0.0

        # Calculate mean and standard deviation along the last dimension
        inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
        inputs_std = mx.std(inputs, axis=-1, keepdims=True)

        # Calculate the cutoff threshold
        cutoff_x = inputs_mean + inputs_std * std_multiplier

        # Apply ReLU to zero out values below the cutoff
        return mx.maximum(0, inputs - cutoff_x)


class Gemma3p5AltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: TextConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.correct_output_scale = mx.zeros(
            (self.config.hidden_size)
        )
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )


    def compute_router_modalities(self, x: mx.array) -> mx.array:
        x_norm: mx.array = self.router_norm(x)
        router_inputs: mx.array = x_norm * self.config.hidden_size**-1.0
        # routed adapted from jax.numpy.einsum("btf,fd->btd", ...)
        routed: mx.array = self.modality_router(router_inputs)
        return mx.tanh(routed)

    def predict(self, x: List[mx.array]) -> List[mx.array]:
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight = mx.clip(self.prediction_coefs.weight, -self.config.altup_coef_clip, self.config.altup_coef_clip)


        # all_coefs adapted from jax.numpy.einsum("...p,pij->...ij", ...)
        all_coefs: mx.array = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(
            *modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs
        )

        outputs: list[mx.array] = [mx.zeros_like(x[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            output = outputs[i]

            for j in range(self.config.altup_num_inputs):
                coef = mx.expand_dims(all_coefs[..., i, j], axis=-1)
                output += coef * x[j]

            x_i = x[i]
            outputs[i] = (x_i + output).astype(x_i.dtype)

        return outputs

    def correct(self, predictions: List[mx.array], activated: mx.array):
        modalities = self.compute_router_modalities(activated)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight = mx.clip(self.correction_coefs.weight, -self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        all_coefs: mx.array = self.correction_coefs(modalities)
        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        corrected = [mx.zeros_like(predictions[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            coef = mx.expand_dims(all_coefs[..., i] + 1, axis=-1)
            corrected[i] = (predictions[i] + coef * innovation).astype(activated.dtype)

        return corrected

    def scale_corrected_output(self, corrected: mx.array):
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale

    def __call__(self, x: List[mx.array], activated: mx.array):
        predictions = self.predict(x)
        corrected = self.correct(predictions=predictions, activated=activated)
        return corrected




class Gemma3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3p5Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = Gemma3p5RMSNorm(
            dim=self.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )
        self.post_attention_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        self.is_sliding = self.self_attn.is_sliding

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = nn.GELU()

        self.altup = Gemma3p5AltUp(config)
        self.laurel = Gemma3p5LaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True)
        self.post_laurel_norm = Gemma3p5RMSNorm(self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False)


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
    ):

        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)


        attn = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / mx.sqrt(mx.array(2.0))

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = mx.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        for i in range(1, len(corrected_predictions)):
            corrected_predictions[i] = corrected_predictions[i] + first_prediction

        return corrected_predictions

class Gemma3p5TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def __call__(self, x: mx.array):
        return super().__call__(x) * mx.array(self.embed_scale, mx.float32).astype(self.weight.dtype)

class Gemma3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0

        self.embed_tokens = Gemma3p5TextScaledWordEmbedding(config.vocab_size, config.hidden_size, embed_scale=config.hidden_size**0.5)
        self.layers = [
            Gemma3p5DecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = Gemma3p5TextScaledWordEmbedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, bias=False
        )

        self.per_layer_projection_norm = Gemma3p5RMSNorm(
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.altup_projections = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]


        self.altup_unembed_projections = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]

        self.norm = Gemma3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True)

        self._per_layer_projection_scale = mx.array(self.hidden_size**-0.5)
        self._per_layer_input_scale = mx.sqrt(mx.array(2.0))

    def __call__(
        self,
        inputs: mx.array=None,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs
    ):
        per_layer_inputs = kwargs.get("per_layer_inputs", None)
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if per_layer_inputs is None and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)

        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        h0 = h

        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        epsilon_tensor = mx.finfo(mx.float16).min

        h: list[mx.array] = [h0] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj: mx.array = self.altup_projections[i - 1](h[i])
            h[i] = altup_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.stack(h, axis=0)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )
            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c, per_layer_input)

         # Per-layer inputs to single output
        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5

        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj = self.altup_unembed_projections[i - 1](h[i])
            h[i] = altup_unemb_proj.astype(h0.dtype)
            new_magnitude = mx.mean(h[i] ** 2, axis=-1, keepdims=True) ** 0.5
            h[i] *= target_magnitude / mx.maximum(new_magnitude, epsilon_tensor)

        h = mx.mean(mx.stack(h), axis=0)

        return self.norm(h)


    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = mx.logical_and(input_ids >= 0, input_ids < self.vocab_size)
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens).reshape(
            *input_ids.shape, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input
        )
        return result

    def project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: Optional[mx.array] = None
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self._per_layer_projection_scale.astype(inputs_embeds.dtype)

        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1], self.config.num_hidden_layers, self.config.hidden_size_per_layer_input
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self._per_layer_input_scale.astype(inputs_embeds.dtype)



class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping
    def __call__(
        self,
        inputs: mx.array=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        out = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        out = self.lm_head(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            weights["language_model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        caches = []


        for i in range(self.config.num_hidden_layers):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window_pattern,
                        keep=0,
                    )
                )
        return caches
