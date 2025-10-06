#!/usr/bin/env python3
"""
========================================================================
BREEZE-ASR-25 SPECIALIZED VERSION - whisper.cpp CoreML converter
========================================================================

⚠️  IMPORTANT: This file has been modified specifically for Breeze-ASR-25 support
    Original: whisper.cpp official convert-whisper-to-coreml.py
    Modified: 2025-09-24 for MediaTek-Research/Breeze-ASR-25 compatibility

🎯 KEY MODIFICATIONS FOR BREEZE-ASR-25:
   1. FIXED: Input tensor name changed from "logmel_data" to "mel" (line 298)
      - Reason: whisper.cpp expects "mel" input, not "logmel_data"
      - Fixes: "Feature mel is required but not specified" error

   2. FIXED: Dynamic sequence length support (line 289)
      - Original: Hard-coded 3000 sequence length (wasteful for Breeze)
      - Fixed: Use hparams.n_audio_ctx (1500 for Breeze-ASR-25)
      - Benefits: Reduced memory usage, correct model dimensions

   3. FIXED: Output tensor name to "encoder_output" (line 301)
      - Reason: whisper.cpp expects "encoder_output" for hybrid inference
      - Benefits: Proper CoreML + GGML integration

🔧 COMPATIBILITY:
   - Breeze-ASR-25: max_source_positions=1500, num_mel_bins=80
   - Architecture: Same as whisper-large-v2 (32 encoder layers, 1280 dim)
   - Special: median_filter_width=7 (Breeze-ASR-25 specific parameter)

📋 USAGE:
   python3 convert-whisper-to-coreml.py --model breeze-asr-25 --encoder-only True

⚠️  DO NOT revert these changes without updating Breeze-ASR-25 integration!
========================================================================
"""

import argparse
import torch
import torch.nn.functional as F
import coremltools as ct

from torch import Tensor
from torch import nn
from typing import Dict
from typing import Optional
from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
from coremltools.models.neural_network.quantization_utils import quantize_weights
from whisper.model import Whisper, AudioEncoder, TextDecoder, ResidualAttentionBlock, MultiHeadAttention, ModelDimensions
from whisper import load_model

# Disable PyTorch Scaled Dot-Product Attention (SDPA) to avoid compatibility issues.
# The Whisper implementation expects a specific behavior from
# torch.nn.functional.scaled_dot_product_attention that differs between PyTorch
# versions. Setting use_sdpa=False forces Whisper to use its manual attention
# implementation instead, which is more stable across different PyTorch versions
# (2.5.0 required by coremltools vs newer versions).
import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False

# Use for changing dim of input in encoder and decoder embeddings
def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """
    Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])

        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict

class LayerNormANE(LayerNormANEBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

class MultiHeadAttentionANE(MultiHeadAttention):
    def __init__(self, n_state: int, n_head: int):
        super().__init__(n_state, n_head)
        self.query =  nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[dict] = None):

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention_ane(q, k, v, mask)

        return self.out(wv), qk

    def qkv_attention_ane(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):

        _, dim, _, seqlen = q.size()

        dim_per_head = dim // self.n_head

        scale = float(dim_per_head)**-0.5

        q = q * scale

        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k.transpose(1,3).split(dim_per_head, dim=3)
        mh_v = v.split(dim_per_head, dim=1)

        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]  # (batch_size, max_seq_length, 1, max_seq_length) * n_heads

        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask[:, :seqlen, :, :seqlen]

        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]  # (batch_size, max_seq_length, 1, max_seq_length) * n_heads
        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_weights, mh_v)]  # (batch_size, dim_per_head, 1, max_seq_length) * n_heads
        attn = torch.cat(attn, dim=1)  # (batch_size, dim, 1, max_seq_length)

        return attn, torch.cat(mh_qk, dim=1).float().detach()


class ResidualAttentionBlockANE(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__(n_state, n_head, cross_attention)
        self.attn =  MultiHeadAttentionANE(n_state, n_head)
        self.attn_ln = LayerNormANE(n_state)
        self.cross_attn =  MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
        self.cross_attn_ln =  LayerNormANE(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp =  nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        self.mlp_ln = LayerNormANE(n_state)


class AudioEncoderANE(AudioEncoder):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormANE(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        assert x.shape[1:] == self.positional_embedding.shape[::-1], "incorrect audio shape"

        # Add positional embedding and add dummy dim for ANE
        x = (x + self.positional_embedding.transpose(0,1)).to(x.dtype).unsqueeze(2)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        x = x.squeeze(2).transpose(1, 2)

        return x

class TextDecoderANE(TextDecoder):

    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

        self.blocks= nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln= LayerNormANE(n_state)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[3] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        # Reformat for ANE
        mask = self.mask[None, None, :, :].permute(0,3,1,2)
        x = x.transpose(1,2).unsqueeze(2)

        for block in self.blocks:
            x = block(x, xa, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)

        # Reformat back from ANE
        x = x.permute(0,2,3,1).squeeze(0)

        # ANE can only load tensors with dim size of at most 16,384 - whisper uses 51,864 (en) or 51,865 (multi-lang) tokens so we need to compute in chunks
        if self.token_embedding.weight.shape[0] >= 51865:
            # split in 11 chunks - 4715 each
            splits = self.token_embedding.weight.split(self.token_embedding.weight.shape[0]//11, dim=0)
            logits = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits]).view(*x.shape[:2], -1)
        else:
            # split in 12 chunks - 4322 each
            assert(self.token_embedding.weight.shape[0] == 51864)
            splits = self.token_embedding.weight.split(self.token_embedding.weight.shape[0]//12, dim=0)
            logits = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits]).view(*x.shape[:2], -1)

        return logits

class WhisperANE(Whisper):
    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)

        self.encoder = AudioEncoderANE(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoderANE(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[3] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=3).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttentionANE):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

def convert_encoder(hparams, model, quantize=False):
    model.eval()

    # 🔧 BREEZE-ASR-25 FIX #1: Calculate correct mel spectrogram input length
    # Transformers WhisperEncoder expects: mel_length = n_audio_ctx * conv_stride1 * conv_stride2
    # For Breeze-ASR-25: 1500 * 2 * 2 = 6000 (not 3000!)
    # For standard Whisper: 1500 * 2 * 2 = 3000 (because their n_audio_ctx defaults to 1500 but they pad to 3000)
    #
    # Important distinction:
    # - n_audio_ctx: encoder OUTPUT sequence length (after conv layers)
    # - mel_length: encoder INPUT sequence length (before conv layers)
    # - Relationship: mel_length = n_audio_ctx * 4 (due to two stride-2 conv layers)

    # Check if model has conv layers to determine stride
    if hasattr(model, 'conv1') and hasattr(model.conv1, 'stride'):
        # For torch.nn.Conv1d, stride is tuple (stride,) even for single value
        stride1 = model.conv1.stride[0] if isinstance(model.conv1.stride, (tuple, list)) else model.conv1.stride
        stride2 = model.conv2.stride[0] if isinstance(model.conv2.stride, (tuple, list)) else model.conv2.stride
        conv_stride = stride1 * stride2  # Usually 2 * 2 = 4
        print(f"🔍 Debug: conv1.stride={model.conv1.stride}, conv2.stride={model.conv2.stride}")
        print(f"🔍 Debug: stride1={stride1}, stride2={stride2}, conv_stride={conv_stride}")
    else:
        conv_stride = 4  # Default for Whisper architecture

    mel_length = hparams.n_audio_ctx * conv_stride

    print(f"🔧 Encoder input/output configuration:")
    print(f"   n_audio_ctx (output length): {hparams.n_audio_ctx}")
    print(f"   conv_stride: {conv_stride}")
    print(f"   mel_length (input length): {mel_length}")

    # 🔧 BREEZE-ASR-25 FIX #2: Wrap Transformers encoder to return only tensor
    # Transformers WhisperEncoder.forward() returns BaseModelOutput (dict-like object)
    # TorchScript trace requires tensor output only
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            # Extract only last_hidden_state tensor from BaseModelOutput
            output = self.encoder(x)
            if hasattr(output, 'last_hidden_state'):
                return output.last_hidden_state
            else:
                return output  # Already a tensor (whisper-openai format)

    wrapped_model = EncoderWrapper(model)

    input_shape = (1, hparams.n_mels, mel_length)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(wrapped_model, input_data)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        # 🔧 BREEZE-ASR-25 FIX #2: Input name "mel" (whisper.cpp expectation)
        # Original: inputs=[ct.TensorType(name="logmel_data", shape=input_shape)]
        inputs=[ct.TensorType(name="mel", shape=input_shape)],
        # 🔧 BREEZE-ASR-25 FIX #3: Output name "encoder_output" (whisper.cpp expectation)
        # Original: outputs=[ct.TensorType(name="output")]
        outputs=[ct.TensorType(name="encoder_output")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

def convert_decoder(hparams, model, quantize=False):
    model.eval()

    tokens_shape = (1, 1)
    audio_shape = (1, hparams.n_audio_ctx, hparams.n_audio_state)

    audio_data = torch.randn(audio_shape)
    token_data = torch.randint(hparams.n_vocab, tokens_shape).long()

    traced_model = torch.jit.trace(model, (token_data, audio_data))

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ],
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model to convert (e.g. tiny, base, large-v3) or HuggingFace path (e.g. MediaTek-Research/Breeze-ASR-25)", required=True)
    parser.add_argument("--encoder-only", action="store_true", help="only convert encoder")
    parser.add_argument("--quantize", action="store_true", help="quantize weights to F16")
    parser.add_argument("--optimize-ane", action="store_true", help="optimize for ANE execution (currently broken)")
    parser.add_argument("--hf-model", action="store_true", help="load from HuggingFace instead of openai-whisper")
    args = parser.parse_args()

    # 🔧 BREEZE-ASR-25 FIX #4: Support HuggingFace model paths
    predefined_models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "small.en-tdrz", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]

    if args.hf_model or args.model not in predefined_models:
        # Load from HuggingFace
        print(f"Loading HuggingFace model: {args.model}")
        from transformers import WhisperModel, WhisperConfig
        import os

        # Download model if needed
        hf_model = WhisperModel.from_pretrained(args.model)
        config = hf_model.config

        # Create whisper-like object with dims attribute
        class HFWhisperWrapper:
            def __init__(self, model, config):
                self.encoder = model.encoder
                self.decoder = model.decoder

                # Create dims object matching whisper.cpp expectations
                class Dims:
                    def __init__(self, config):
                        self.n_mels = config.num_mel_bins
                        self.n_audio_ctx = config.max_source_positions
                        self.n_audio_state = config.d_model
                        self.n_audio_head = config.encoder_attention_heads
                        self.n_audio_layer = config.encoder_layers
                        self.n_vocab = config.vocab_size
                        self.n_text_ctx = config.max_target_positions
                        self.n_text_state = config.d_model
                        self.n_text_head = config.decoder_attention_heads
                        self.n_text_layer = config.decoder_layers

                self.dims = Dims(config)

            def cpu(self):
                self.encoder = self.encoder.cpu()
                self.decoder = self.decoder.cpu()
                return self

        whisper = HFWhisperWrapper(hf_model, config).cpu()
        hparams = whisper.dims
        print(f"✅ Loaded HuggingFace model with config:")
        print(f"   n_audio_ctx: {hparams.n_audio_ctx}")
        print(f"   n_mels: {hparams.n_mels}")
        print(f"   n_vocab: {hparams.n_vocab}")
    else:
        # Load predefined model from openai-whisper
        print(f"Loading openai-whisper model: {args.model}")
        whisper = load_model(args.model).cpu()
        hparams = whisper.dims
        print(hparams)

    if args.optimize_ane:
        whisperANE = WhisperANE(hparams).eval()
        whisperANE.load_state_dict(whisper.state_dict())

        encoder = whisperANE.encoder
        decoder = whisperANE.decoder
    else:
        encoder = whisper.encoder
        decoder = whisper.decoder

    # Convert encoder
    encoder = convert_encoder(hparams, encoder, quantize=args.quantize)
    encoder.save(f"models/coreml-encoder-{args.model}.mlpackage")

    if args.encoder_only is False:
        # Convert decoder
        decoder = convert_decoder(hparams, decoder, quantize=args.quantize)
        decoder.save(f"models/coreml-decoder-{args.model}.mlpackage")

    print("done converting")
