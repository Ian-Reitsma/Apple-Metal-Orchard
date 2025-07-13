#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchard FlashAttention Monkeypatch — Debug Edition
Import before model load; monkeypatches only if USE_FLASH_ATTN=1.
Captures rich context for every kernel call and logs all patch diagnostics.
"""

import os, sys, torch, traceback
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

PATCH_ENABLED = os.getenv("USE_FLASH_ATTN") == "1"
LOG_KERNEL_CALLS = bool(int(os.getenv("FLASHATTN_LOG_KERNEL", "1")))
LOG_KERNEL_DETAIL = bool(int(os.getenv("FLASHATTN_LOG_DETAIL", "0")))
KERNEL_LOG_PATH = "/tmp/flashattn_kernel_calls.log"

kernel_call_count = 0
patch_context_logged = False

def _summarize_tensor(t, name):
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"min={t.min().item() if t.numel()>0 else 'n/a'}, "
            f"max={t.max().item() if t.numel()>0 else 'n/a'}, "
            f"nan={torch.isnan(t).any().item() if t.dtype.is_floating_point else False}")

if PATCH_ENABLED:
    try:
        _original_forward = GPT2Attention.forward

        def flashattention_forward(
            self,
            hidden_states,
            past_key_value=None,
            cache_position=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            **kwargs
        ):
            global kernel_call_count, patch_context_logged

            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention:
                query_states = self.q_attn(hidden_states)
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                attention_mask = encoder_attention_mask
            else:
                query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

            shape_q = (*query_states.shape[:-1], -1, self.head_dim)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            query_states = query_states.view(shape_q).transpose(1, 2)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

            kernel_call_count += 1

            # On first call, log detailed patch context for debugging
            if not patch_context_logged:
                patch_context_logged = True
                try:
                    print("[orchard][flashpatch] --- FlashAttention Patch Context ---", file=sys.stderr)
                    print(f"  PATCH_ENABLED: {PATCH_ENABLED}", file=sys.stderr)
                    print(f"  FLASHATTN_LOG_KERNEL: {LOG_KERNEL_CALLS}", file=sys.stderr)
                    print(f"  FLASHATTN_LOG_DETAIL: {LOG_KERNEL_DETAIL}", file=sys.stderr)
                    print(f"  torch version: {torch.__version__}", file=sys.stderr)
                    print(f"  Model config (first layer): {getattr(self, 'config', 'n/a')}", file=sys.stderr)
                    print(f"  Example Q shape: {query_states.shape}, dtype: {query_states.dtype}, device: {query_states.device}", file=sys.stderr)
                    print(f"  ENV: USE_FLASH_ATTN={os.getenv('USE_FLASH_ATTN')}, PYTHONHASHSEED={os.getenv('PYTHONHASHSEED')}", file=sys.stderr)
                    print("-------------------------------------------------------------", file=sys.stderr)
                except Exception as err:
                    print(f"[orchard][flashpatch] WARN: Failed to log patch context: {err}", file=sys.stderr)

            if LOG_KERNEL_CALLS:
                try:
                    logline = (f"flashattn_call {kernel_call_count}: "
                               f"layer={getattr(self, 'layer_idx', '?')}, "
                               f"q={tuple(query_states.shape)}, k={tuple(key_states.shape)}, v={tuple(value_states.shape)}\n")
                    with open(KERNEL_LOG_PATH, "a") as f:
                        f.write(logline)
                    if LOG_KERNEL_DETAIL:
                        with open(KERNEL_LOG_PATH, "a") as f:
                            f.write("    " + _summarize_tensor(query_states, "Q") + "\n")
                            f.write("    " + _summarize_tensor(key_states, "K") + "\n")
                            f.write("    " + _summarize_tensor(value_states, "V") + "\n")
                except Exception as e:
                    print(f"[orchard][flashpatch] WARN: could not log kernel call ({e})", file=sys.stderr)

            scale = 1.0 / (self.head_dim ** 0.5)
            causal = attention_mask is None
            try:
                attn_output = torch.ops.flash_attn_mps._flash_attn_fwd(
                    query_states, key_states, value_states, scale, causal
                )
                print(f"[DEBUG][FLASHATTN] Output shape after kernel: {attn_output.shape}, dtype: {attn_output.dtype}")
                attn_weights = None
            except Exception as e:
                print(f"[orchard][flashpatch] ERROR: FlashAttention kernel failed on call {kernel_call_count} (layer={getattr(self, 'layer_idx', '?')}): {e}", file=sys.stderr)
                print("---- Kernel Input Shapes ----", file=sys.stderr)
                print(f"  Q: {query_states.shape}, dtype={query_states.dtype}, device={query_states.device}", file=sys.stderr)
                print(f"  K: {key_states.shape}, dtype={key_states.dtype}, device={key_states.device}", file=sys.stderr)
                print(f"  V: {value_states.shape}, dtype={value_states.dtype}, device={value_states.device}", file=sys.stderr)
                print(f"[DEBUG][Q] {q.shape} {q.dtype} {q.device}")
                print(f"[DEBUG][K] {k.shape} {k.dtype} {k.device}")
                print(f"[DEBUG][V] {v.shape} {v.dtype} {v.device}")
                print("---- Kernel Exception Trace ----", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("[orchard][flashpatch] Fallback to original GPT2Attention.forward", file=sys.stderr)
                return _original_forward(
                    self,
                    hidden_states,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    **kwargs
                )

            attn_output = attn_output.permute(0, 2, 1, 3)
            print(f"[DEBUG][AFTER PERMUTE] {attn_output.shape}")
            batch, seq, n_heads, head_dim = attn_output.shape
            attn_output = attn_output.reshape(batch, seq, n_heads * head_dim).contiguous()
            print(f"[DEBUG][AFTER RESHAPE] {attn_output.shape}")
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, attn_weights

        GPT2Attention.forward = flashattention_forward
        print(f"[orchard][flashpatch] FlashAttention monkeypatch ENABLED (GPT2Attention.forward, log={LOG_KERNEL_CALLS}, detail={LOG_KERNEL_DETAIL})", file=sys.stderr)
    except Exception as e:
        print(f"[orchard][flashpatch] ERROR: Could not patch GPT2Attention.forward: {e}", file=sys.stderr)
        PATCH_ENABLED = False
else:
    print("[orchard][flashpatch] FlashAttention monkeypatch NOT ENABLED (USE_FLASH_ATTN!=1)", file=sys.stderr)
