"""
Optional: route attention through SageAttention (8-bit QK, e.g. SageAttention-rocm on AMD GPUs) instead of
torch.nn.functional.scaled_dot_product_attention. Call use_sage_attention() once before running the pipeline.

Only unmasked, non-causal fp16/bf16 calls with head dim 64 or 128, equal query/value head dims and at least 128
queries go to SageAttention; everything else keeps PyTorch's kernel.
"""

import torch
import torch.nn.functional as F


def use_sage_attention() -> bool:
    try:
        from sageattention import sageattn
    except ImportError:
        return False
    if getattr(F.scaled_dot_product_attention, "_sage", False):
        return True
    sdpa = F.scaled_dot_product_attention

    def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (attn_mask is None and not is_causal and dropout_p == 0.0 and not kwargs
                and q.dtype in (torch.float16, torch.bfloat16) and q.dim() == 4 and q.shape[-1] in (64, 128)
                and v.shape[-1] == q.shape[-1] and q.shape[-2] >= 128):
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=scale)
        return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    scaled_dot_product_attention._sage = True
    F.scaled_dot_product_attention = scaled_dot_product_attention
    return True
