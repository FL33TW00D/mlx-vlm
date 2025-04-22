from dataclasses import dataclass
from typing import Callable, Optional, OrderedDict, Tuple, Union
from collections.abc import Sequence

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


# (x: mx.array, mask: mx.BoolArray (no BoolArray in mlx))
type SLSequence = Tuple[mx.array, mx.array]


class SequenceLayer(nn.Module):
    layers: Callable[[SLSequence], SLSequence]

    def __call__(self, x: SLSequence) -> SLSequence:
        return self.layers(x)


class SequenceLayerConv2d(SequenceLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        *args,
        padding: tuple[int, int] = (0, 0),
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride

        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias,
        )

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = self.conv(y)
        return y, mask


class SequenceLayerDense(SequenceLayer):
    def __init__(self, shape: tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shape = shape
        self.weight = mx.empty(self.shape)

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = mx.einsum("...a,ab->...b", y, self.weight)
        return y, mask


class SequenceLayerDenseShaped(SequenceLayer):
    def __init__(
        self,
        *args,
        input_shape: Sequence[int] = (),
        output_shape: Sequence[int] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_shape = tuple(input_shape)
        self.input_dims = "".join(
            chr(ord("a") + i) for i in range(len(self.input_shape))
        )
        self.input_weight_shape = self.input_shape or (1,)
        self.input_weight_dims = self.input_dims or "I"
        self.output_shape = tuple(output_shape)
        self.output_dims = "".join(
            chr(ord("a") + i + len(self.input_shape))
            for i in range(len(self.output_shape))
        )
        self.output_weight_shape = self.output_shape or (1,)
        self.output_weight_dims = self.output_dims or "O"
        self.equation = f"BT{self.input_dims},{self.input_weight_dims}{self.output_weight_dims}->BT{self.output_dims}"

        weight_shape = self.input_weight_shape + self.output_weight_shape
        self.weight = mx.empty(weight_shape)

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = mx.einsum(self.equation, y, self.weight)
        return y, mask


class SequenceLayerDepthwiseConv1D(SequenceLayer):
    pass


class SequenceLayerExpandDims(SequenceLayer):
    def __init__(self, dims: Union[int, Sequence[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (dims,) if isinstance(dims, int) else dims

    def _normalize_dims(
        self,
        x_ndims: int,
    ) -> Sequence[int]:
        dims = [d + x_ndims if d < 0 else d for d in self.dims]
        dims = sorted(dims)
        for d in dims:
            if d < 0 or d > x_ndims:
                raise ValueError(f"Received invalid dim for expansion: {d}")
        return dims

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y_dims = self._normalize_dims(y.ndim)
        for d in y_dims:
            y = mx.expand_dims(y, axis=d)
        return y, mask


class SequenceLayerGatedLinearUnit(SequenceLayer):
    def __call__(self, x: SLSequence) -> SLSequence:
        x, mask = x
        feature, gate = mx.split(x, 2, dim=-1)
        gate = mx.sigmoid(gate)
        x = feature * gate
        return x, mask


class SequenceLayerGroupNorm(SequenceLayer):
    def __init__(
        self, num_groups: int, num_channels: int, *args, eps: float = 1e-3, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.norm = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.num_channels, eps=self.eps
        )

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = self.norm(y)
        return y, mask


class SequenceLayerLocalDotProductSelfAttention(SequenceLayer):
    pass


class SequenceLayerMaskInvalid(SequenceLayer):
    pass


class SequenceLayerRelu(SequenceLayer):
    def __call__(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = nn.relu(x)
        return x, mask


class SequenceLayerResidual(SequenceLayer):
    def __init__(
        self,
        layers: nn.Sequential,
        *args,
        shortcut_layers: Optional[nn.Sequential] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.shortcut_layers = shortcut_layers

    def residual_function(self, x: SLSequence, shortcut_x: SLSequence) -> SLSequence:
        y = x[0] + shortcut_x[0]
        mask = x[1] | shortcut_x[1]
        return y, mask

    def __call__(self, x: SLSequence) -> SLSequence:
        y: SLSequence = self.layers(x)
        if self.shortcut_layers is not None:
            shortcut_y: SLSequence = self.shortcut_layers(x)
            y = self.residual_function(y, shortcut_y)
        return y


class SequenceLayerRMSNorm(SequenceLayer):
    def __init__(
        self, shape: Sequence[int], *args, dim: int = -1, eps: float = 1e-6, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.shape = shape

        self.dim = dim
        self.eps = eps

        self.scale = mx.ones(self.shape)

    def forward(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y_dtype = y.dtype
        y = y.float()
        mean_squared = y.pow(2).mean(dim=self.dim, keepdim=True)
        root_mean_squared = y * mx.rsqrt(mean_squared + self.eps)
        scaled = root_mean_squared * self.scale.float()
        return scaled.type(y_dtype), mask


class SequenceLayerScale(SequenceLayer):
    def __init__(self, factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = x * self.factor
        return x, mask


class SequenceLayerSwish(SequenceLayer):
    def forward(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = nn.functional.silu(x)
        return x, mask


class SequenceLayerTransformerXLRelativePositionEmbedding(SequenceLayer):
    pass


class Gemma3p5AudioSSCPConvBlock(SequenceLayer):
    def __init__(self, config: AudioConfig, idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.out_channels = self.config.sscp_conv_channel_size[idx]
        self.kernel_size = self.config.sscp_conv_kernel_size[idx]
        self.stride = self.config.sscp_conv_stride_size[idx]

        # input_channels is equal to either the out_channels from the prior
        # Conv2d or 1 if this is the first Conv2d.
        if idx > 0:
            self.input_channels = self.config.sscp_conv_channel_size[idx - 1]
        else:
            self.input_channels = 1

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv2d",
                        SequenceLayerConv2d(
                            in_channels=self.input_channels,
                            out_channels=self.out_channels,
                            kernel=self.kernel_size,
                            stride=self.stride,
                        ),
                    ),
                    (
                        "norm",
                        SequenceLayerGroupNorm(
                            num_groups=1, num_channels=self.out_channels
                        ),
                    ),
                    ("relu", SequenceLayerRelu()),
                ]
            )
        )


class Gemma3p5AudioSubSampleConvProjection(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("expand", SequenceLayerExpandDims(dims=-1)),
                    ("conv_0", Gemma3p5AudioSSCPConvBlock(config, 0)),
                    ("conv_1", Gemma3p5AudioSSCPConvBlock(config, 1)),
                    (
                        "input_proj",
                        SequenceLayerDenseShaped(
                            input_shape=(
                                self.config.sscp_conv_channel_size[1],
                                self.config.sscp_conv_channel_size[1],
                            ),
                            output_shape=(self.config.hidden_size,),
                        ),
                    ),
                ]
            )
        )


class Gemma3p5AudioConformerAttention(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.embedding = SequenceLayerTransformerXLRelativePositionEmbedding()
        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(
                OrderedDict(
                    [
                        (
                            "pre_attn_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "attn",
                            SequenceLayerLocalDotProductSelfAttention(
                                num_attention_heads=self.config.conf_num_attention_heads,
                                attention_head_size=self.config.conf_attention_chunk_size,
                                attention_logits_soft_cap=self.config.conf_attention_logit_cap,
                            ),
                        ),
                        (
                            "post_attn_dense",
                            SequenceLayerDenseShaped(
                                input_shape=(
                                    self.config.conf_num_attention_heads,
                                    self.config.hidden_size
                                    // self.config.conf_num_attention_heads,
                                ),
                                output_shape=(self.config.hidden_size,),
                            ),
                        ),
                        (
                            "post_attn_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                    ]
                )
            )
        )


class Gemma3p5AudioConformerFeedForward(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(
                OrderedDict(
                    [
                        (
                            "pre_layer_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "ffw_layer_1",
                            SequenceLayerDense(
                                shape=(
                                    self.config.hidden_size,
                                    self.config.hidden_size * 4,
                                )
                            ),
                        ),
                        (
                            "ffw_layer_2",
                            SequenceLayerDense(
                                shape=(
                                    self.config.hidden_size * 4,
                                    self.config.hidden_size,
                                )
                            ),
                        ),
                        (
                            "post_layer_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "post_layer_scale",
                            SequenceLayerScale(factor=config.conf_residual_weight),
                        ),
                    ]
                )
            )
        )


class Gemma3p5AudioConformerLightConv1d(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(
                OrderedDict(
                    [
                        (
                            "pre_layer_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "linear_start",
                            SequenceLayerDense(
                                shape=(
                                    self.config.hidden_size,
                                    self.config.hidden_size * 2,
                                )
                            ),
                        ),
                        ("glu", SequenceLayerGatedLinearUnit()),
                        (
                            "depthwise_conv1d",
                            SequenceLayerDepthwiseConv1D(
                                kernel_size=self.config.conf_conv_kernel_size,
                                strides=1,
                                output_channels=self.config.hidden_size,
                            ),
                        ),
                        (
                            "conv_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        ("conv_activation", SequenceLayerSwish()),
                        (
                            "linear_end",
                            SequenceLayerDense(
                                shape=(self.config.hidden_size, self.config.hidden_size)
                            ),
                        ),
                    ]
                )
            )
        )


class Gemma3p5AudioConformerBlock(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("ffw_layer_start", Gemma3p5AudioConformerFeedForward(self.config)),
                    ("attention", Gemma3p5AudioConformerAttention(self.config)),
                    ("lconv1d", Gemma3p5AudioConformerLightConv1d(self.config)),
                    ("ffw_layer_end", Gemma3p5AudioConformerFeedForward(self.config)),
                    ("norm", SequenceLayerRMSNorm(shape=(self.config.hidden_size,))),
                ]
            )
        )


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

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("subsample_conv_projection", self.subsample_conv_projection),
                    ("conformer", self.conformer_blocks),
                    ("reducer", self.uniform_reducer),
                    ("mask_invalid", SequenceLayerMaskInvalid()),
                ]
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError()
