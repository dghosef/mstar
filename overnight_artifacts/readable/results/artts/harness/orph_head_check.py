#!/usr/bin/env python3
"""Greedy rollouts with the HF Orpheus checkpoint: at every step, does the
custom-rows-only head (F.pad of weight[base:] logits) pick the token the full
head picks?  Also counts steps whose full-head argmax is a text token."""
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "canopylabs/orpheus-3b-0.1-ft"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16).cuda().eval()
W = model.lm_head.weight
BASE, STOP = 128256, 128258
sentences = [l.strip() for l in open(sys.argv[1]) if l.strip()][: int(sys.argv[2])]
steps = agree = text_argmax = 0
with torch.inference_mode():
    for text in sentences:
        ids = tok(f"tara: {text}", return_tensors="pt").input_ids[0]
        ids = torch.cat([torch.tensor([128259]), ids, torch.tensor([128009, 128260, 128261, 128257])]).cuda()[None]
        out = model(ids, use_cache=True)
        past, h = out.past_key_values, None
        hidden = model.model.norm  # noqa: B018 (hidden states come from output_hidden_states below)
        x = ids
        for _ in range(400):
            o = model(x, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = o.past_key_values
            hl = o.hidden_states[-1][:, -1]
            full = F.linear(hl, W).float()
            sliced = F.pad(F.linear(hl, W[BASE:]), (BASE, 0), value=float("-inf")).float()
            a, b = full.argmax(-1).item(), sliced.argmax(-1).item()
            steps += 1; agree += a == b; text_argmax += a < BASE
            if a == STOP:
                break
            x = torch.tensor([[a]], device="cuda")
print(f"sentences={len(sentences)} steps={steps} argmax_agree={agree} full_argmax_in_text_rows={text_argmax}")
