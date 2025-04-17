from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioConfig:
    input_feat_size: int = 80
    hidden_size: int = 1536
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_positional_bias_size: int = 256
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))


class Gemma3p5AudioSSCPConvBlock(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioSubSampleConvProjection(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioConformerAttention(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioConformerFeedForward(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioConformerInNetorkGraidentClipping(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioConformerLightConv1d(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioConformerBlock(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config


class Gemma3p5AudioUniformReducer(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.reduction_factor = config.conf_reduction_factor

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        if self.reduction_factor > 1:
            x = x[:, :: self.reduction_factor]
            mask = mask[:, :: self.reduction_factor]
        return x, mask


class AudioModel(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.subsample_conv_projection = Gemma3p5AudioSubSampleConvProjection(config)
        self.conformer_blocks = [
            Gemma3p5AudioConformerBlock(config)
            for _ in range(config.conf_num_hidden_layers)
        ]
        self.uniform_reducer = Gemma3p5AudioUniformReducer(config)

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError()
