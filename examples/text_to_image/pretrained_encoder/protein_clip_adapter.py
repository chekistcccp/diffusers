import os

import torch
try:
    import torch_npu
except ImportError:
    pass
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace

from transformers import CLIPTextModel, CLIPTokenizer
from .tokenizer import ProtTokenizer
from .model import EmbeddingFromPretrained, DAN


def get_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def empty_cache():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_causal_attention_mask(seq_length, device, dtype):
    """Create the causal attention mask used by CLIP text encoder.
    Upper triangle is -inf (cannot attend to future tokens), lower triangle is 0.
    Shape: (1, 1, seq_length, seq_length)
    """
    mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask_cond = torch.arange(mask.shape[1], device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.shape[1], 1), 0)
    return mask.unsqueeze(0).unsqueeze(0)


class ProteinCLIPAdapter(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        embeddings_dir,
        vocabulary_path,
        species_classes_path,
        max_tokens=6000,
        num_compressed_tokens=75,
        protein_embedding_dim=1024,
        clip_hidden_dim=768,
        device=None,
    ):
        super().__init__()

        self._device = get_device(device)
        self.max_tokens = max_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.clip_hidden_dim = clip_hidden_dim
        self.protein_embedding_dim = protein_embedding_dim

        clip_text_encoder = None
        text_encoder_dir = os.path.join(pretrained_model_name_or_path, "text_encoder")
        parent_text_encoder_dir = os.path.join(os.path.dirname(pretrained_model_name_or_path), "text_encoder")
        for path in [text_encoder_dir, parent_text_encoder_dir, pretrained_model_name_or_path]:
            if os.path.isdir(path) and any(
                os.path.isfile(os.path.join(path, f))
                for f in ["config.json", "model.safetensors", "pytorch_model.bin"]
            ):
                try:
                    clip_text_encoder = CLIPTextModel.from_pretrained(path)
                    break
                except Exception:
                    continue
        if clip_text_encoder is None:
            raise RuntimeError(
                f"无法从本地加载 CLIP text encoder，"
                f"请确认目录中包含 config.json 和模型权重文件"
            )

        self.clip_config = clip_text_encoder.config
        self.clip_text_model = clip_text_encoder.text_model

        for param in self.clip_text_model.parameters():
            param.requires_grad = False
        self.clip_text_model.eval()

        clip_tokenizer = None
        tokenizer_dir = os.path.join(pretrained_model_name_or_path, "tokenizer")
        parent_dir = os.path.dirname(pretrained_model_name_or_path)
        parent_tokenizer_dir = os.path.join(parent_dir, "tokenizer") if os.path.basename(pretrained_model_name_or_path) == "text_encoder" else None
        for path in [tokenizer_dir, parent_tokenizer_dir, pretrained_model_name_or_path, parent_dir]:
            if path is None or not os.path.isdir(path):
                continue
            if os.path.isfile(os.path.join(path, "tokenizer_config.json")):
                try:
                    clip_tokenizer = CLIPTokenizer.from_pretrained(path)
                    break
                except Exception:
                    continue
        if clip_tokenizer is None:
            raise RuntimeError(
                f"无法从本地加载 CLIP tokenizer，"
                f"请确认目录中包含 tokenizer_config.json"
            )
        self.clip_tokenizer = clip_tokenizer
        self.model_max_length = clip_tokenizer.model_max_length

        vocab_to_index = OrderedDict()
        with open(vocabulary_path, 'r') as f:
            for line in f:
                idx, token = line.strip().split(',', 1)
                vocab_to_index[token] = int(idx)

        self.tokenizer = ProtTokenizer(
            vocab_to_index, max_length=max_tokens, model_max_length=max_tokens
        )

        self.protein_embedding_layer = EmbeddingFromPretrained(
            vector_size=protein_embedding_dim,
            embed_dir=embeddings_dir,
            sequence_max_length=max_tokens,
            device=self._device,
        ).to(self._device)
        for param in self.protein_embedding_layer.parameters():
            param.requires_grad = False

        self.compression_module = AttentionCompressionFixed(
            input_dim=protein_embedding_dim,
            output_dim=clip_hidden_dim,
            num_tokens=max_tokens,
            num_compressed_tokens=num_compressed_tokens,
        )

        self.sos_token = nn.Parameter(torch.randn(1, 1, clip_hidden_dim) * 0.02)
        self.eos_token = nn.Parameter(torch.randn(1, 1, clip_hidden_dim) * 0.02)

        clip_position_embedding = self.clip_text_model.embeddings.position_embedding.weight.data
        self.position_embedding = nn.Embedding(
            self.model_max_length, clip_hidden_dim
        )
        self.position_embedding.weight.data.copy_(clip_position_embedding)
        self.position_embedding.weight.requires_grad = True

        self.config = SimpleNamespace(
            max_tokens=int(max_tokens),
            embeddings=embeddings_dir,
            species_classes=species_classes_path,
            vocabulary=vocabulary_path,
            use_attention_mask=True,
            hidden_size=clip_hidden_dim,
            max_position_embeddings=self.model_max_length,
        )

    def encode_protein(self, input_ids, attention_mask=None):
        x, _ = self.protein_embedding_layer(input_ids)

        compression_dtype = self.compression_module.query_tokens.dtype
        compression_device = self.compression_module.query_tokens.device
        x = x.to(device=compression_device, dtype=compression_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=compression_device)

        compressed = self.compression_module(x, attention_mask=attention_mask)

        batch_size = compressed.shape[0]
        sos = self.sos_token.expand(batch_size, -1, -1)
        eos = self.eos_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([sos, compressed, eos], dim=1)

        seq_len = hidden_states.shape[1]  # 1 + 75 + 1 = 77
        position_ids = torch.arange(seq_len, device=hidden_states.device)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_embeds.unsqueeze(0)

        if seq_len < self.model_max_length:
            pad_len = self.model_max_length - seq_len
            padding = torch.zeros(
                batch_size, pad_len, self.clip_hidden_dim,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, padding], dim=1)
            clip_attention_mask = torch.cat([
                torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.long),
                torch.zeros(batch_size, pad_len, device=hidden_states.device, dtype=torch.long),
            ], dim=1)
        elif seq_len > self.model_max_length:
            hidden_states = hidden_states[:, :self.model_max_length, :]
            clip_attention_mask = torch.ones(
                batch_size, self.model_max_length,
                device=hidden_states.device, dtype=torch.long
            )
        else:
            clip_attention_mask = torch.ones(
                batch_size, self.model_max_length,
                device=hidden_states.device, dtype=torch.long
            )

        return hidden_states, clip_attention_mask

    def forward(self, input_ids, attention_mask=None, return_dict=False, **kwargs):
        target_device = next(self.parameters()).device
        input_ids = input_ids.to(target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        inputs_embeds, clip_attention_mask = self.encode_protein(input_ids, attention_mask)

        clip_dtype = self.clip_text_model.embeddings.position_embedding.weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=clip_dtype)

        # Build causal attention mask (required by CLIP text encoder)
        # CLIP uses causal attention: each token can only attend to itself and preceding tokens
        seq_length = inputs_embeds.shape[1]
        causal_attention_mask = _create_causal_attention_mask(
            seq_length, inputs_embeds.device, clip_dtype
        )

        # Convert clip_attention_mask to the 4D format expected by CLIP encoder
        # From (batch, seq_len) -> (batch, 1, seq_len, seq_len)
        if clip_attention_mask is not None:
            # expand to 4D: (batch, 1, 1, seq_len) -> broadcast over (batch, heads, seq_len, seq_len)
            extended_attention_mask = clip_attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=clip_dtype)
            # 1.0 for tokens to attend to, 0.0 for masked tokens -> convert to additive mask
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(clip_dtype).min
        else:
            extended_attention_mask = None

        clip_outputs = self.clip_text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        last_hidden_state = self.clip_text_model.final_layer_norm(clip_outputs.last_hidden_state)

        # Pooled output: take the EOS token (position seq_len-1 = 76 for 77 tokens)
        # This matches CLIP's convention where the last valid token is used for pooling
        eos_token_index = clip_attention_mask.sum(dim=1) - 1  # (batch,)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            eos_token_index,
        ]

        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPooling
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
            )
        return (last_hidden_state, pooled_output)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


class AttentionCompressionFixed(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens, num_compressed_tokens):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_compressed_tokens = num_compressed_tokens

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_compressed_tokens, input_dim) * 0.02
        )
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.dim_compress = nn.Linear(input_dim, output_dim)

        self.position_embedding = nn.Parameter(
            torch.randn(1, num_compressed_tokens, input_dim) * 0.02
        )

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]

        query = (self.query_tokens + self.position_embedding).expand(batch_size, -1, -1)
        key = self.key_proj(x)
        value = self.value_proj(x)

        scale = self.input_dim ** -0.5
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * scale

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                valid_tokens = attention_mask.to(dtype=torch.bool)
            else:
                valid_tokens = attention_mask
            valid_tokens = valid_tokens.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~valid_tokens, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1)
        compressed = torch.bmm(attn_weights, value)

        output = self.dim_compress(compressed)
        return output
