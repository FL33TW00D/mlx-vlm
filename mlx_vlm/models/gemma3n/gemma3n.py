import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .audio import AudioModel, Gemma3nAudioEmbedder
from .language import LanguageModel, TextConfig
from .vision import Gemma3p5VisionEmbedder, VisionConfig, VisionModel
from .config import ModelConfig

def masked_scatter(input_tensor, mask, source):
    """MLX implementation of PyTorch's masked_scatter - simplified version"""
    mask = mask.astype(mx.bool_)
    result = mx.broadcast_to(input_tensor, mask.shape).flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    # Early return if no values to scatter
    if not mask_flat.any():
        return result.reshape(mask.shape)

    # Create indices for source values using cumsum
    # This gives us 0, 1, 2, ... for True positions in mask
    source_indices = mx.cumsum(mask_flat.astype(mx.int32)) - 1

    # Clamp indices to source bounds
    source_indices = mx.clip(source_indices, 0, len(source_flat) - 1)

    # Select source values for each position
    selected_values = source_flat[source_indices]

    # Use where to scatter: if mask is True, use source value, else original
    result = mx.where(mask_flat, selected_values, result)

    return result.reshape(mask.shape)

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config


        # Text
        self.language_model = LanguageModel(config.text_config)

        # Vision
        # self.vision_tower = VisionModel(config.vision_config)
        # self.embed_vision = Gemma3p5VisionEmbedder(config.vision_config)

        # # Audio
        audio_vocab_offset = config.text_config.vocab_size + config.vision_config.vocab_size

        self.audio_tower = AudioModel(config.audio_config)
        self.embed_audio = Gemma3nAudioEmbedder(config, vocab_offset=audio_vocab_offset)

    def embed(self, input_ids):
        text_input_ids = mx.where(input_ids < self.config.vocab_size, input_ids, 0)

        inputs_embeds = self.language_model.model.embed_tokens(text_input_ids)

        audio_embeds = self.embed_audio(input_ids=input_ids)
        inputs_embeds = mx.where(
            input_ids[..., None] < self.embed_audio.vocab_offset,
            inputs_embeds,
            audio_embeds,
        )
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None and input_features is None:
            return self.embed(input_ids)

        inputs_embeds = self.embed(input_ids)

        if input_features is not None:
            audio_outputs, audio_mask = self.get_audio_features(input_features, ~input_features_mask)
            padding_tok = mx.array([[self.config.text_config.pad_token_id]])
            padding_embs = self.embed_audio(input_ids=padding_tok)

            audio_outputs = mx.where(audio_mask[..., None], padding_embs, audio_outputs)

            extra_padding_tokens = self.config.audio_soft_tokens_per_image - audio_outputs.shape[1]
            extra_padding_features = mx.broadcast_to(
                padding_embs, (audio_outputs.shape[0], extra_padding_tokens, padding_embs.shape[2])
            )


            audio_outputs = mx.concatenate((audio_outputs, extra_padding_features), axis=1)
            return self.merge_multimodal_and_text(
                input_ids, inputs_embeds, audio_outputs, self.config.audio_token_id, modality="audio"
            )

    def get_audio_features(self, input_features, input_features_mask):
        audio_outputs, audio_mask = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask

    def get_image_features(self, pixel_values):
        vision_outputs, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).transpose(0, 2, 1)

        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(vision_outputs, is_soft_embedding=True)

    def merge_multimodal_and_text(self, input_ids, inputs_embeds, features, token_id, modality="image"):

        if input_ids is None:
            special_modality_mask = inputs_embeds == self.embed_audio(
                input_ids=mx.array([self.config.audio_token_id])
            )
        else:
            special_modality_mask = mx.expand_dims(input_ids == token_id, -1)
            special_modality_mask = mx.broadcast_to(
                special_modality_mask, inputs_embeds.shape
            )

        # Count special tokens by summing the mask
        modality_tokens_in_text = special_modality_mask.sum()
        feature_tokens = features.size

        if modality_tokens_in_text != feature_tokens:
            raise ValueError(
                f"Number of {modality}s does not match number of special {modality} tokens in the input text. "
                f"Got {modality_tokens_in_text} {modality} tokens in the text and "
                f"{feature_tokens} tokens from {modality} embeddings."
            )
        features = features.astype(inputs_embeds.dtype)

        inputs_embeds = masked_scatter(inputs_embeds, special_modality_mask, features)
        return inputs_embeds


    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        # Audio features
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, input_features=input_features, input_features_mask=input_features_mask, **kwargs
        )

        logits = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )
        return logits

    def sanitize(self, weights):
        sanitized_weights = {".".join(k.split(".")[1:]): v for k, v in weights.items() if "vision_tower" not in k and "embed_vision" not in k}
        return sanitized_weights

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model
