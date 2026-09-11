"""
Model loading and residual-stream activation extraction (torch).

Moved verbatim from snr_sweep.py.

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
import torch


def get_model(name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, output_hidden_states=True).to(device).eval()
    return tok, model


@torch.no_grad()
def extract_all_layers(statements, tok, model, device, batch_size=16):
    """Last-token residual activations for every layer in one pass per batch.

    Returns float32 array of shape (n_layers+1, N, d). Padding is right-side,
    so the final real token is found from the attention mask.
    """
    chunks = []
    for i in range(0, len(statements), batch_size):
        enc = tok(statements[i:i + batch_size], return_tensors="pt",
                  padding=True, truncation=True, max_length=128).to(device)
        hs = model(**enc).hidden_states           # (n_layers+1) tensors (B,T,d)
        last = enc["attention_mask"].sum(1) - 1   # index of final real token
        b = torch.arange(last.shape[0], device=device)
        batch = torch.stack([h[b, last] for h in hs])    # (L, B, d)
        chunks.append(batch.float().cpu().numpy())
        del hs, enc, batch
    return np.concatenate(chunks, axis=1)
