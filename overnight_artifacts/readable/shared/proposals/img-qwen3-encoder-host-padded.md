# img: Qwen3 diffusion text encoder host-side `padded` flag + cached rotary tables

- branch perf/ovn-img, commit 2a4e1d9bb (the shared part is `mstar/model/components/diffusion/qwen3/encoder.py`; the rest is flux2_klein model code)
- Mechanism: `Qwen3HiddenStateEncoder.forward(ids, mask, padded=None)`. When the caller knows on the host whether the batch is padded, the `bool(mask.all())` host sync is skipped. The rotary cos/sin tables are cached per (seq, device), since building them does a pageable H2D of inv_freq. Together these make the encoder graph-capturable. `padded=None` keeps the old behaviour exactly.
- Users: only flux2_klein and z_image, both of which belong to img. Bit-identical outputs (served klein images identical to base, 8/8 prompts, 2 rounds).
- Numbers (klein text encoder captured, H200): c=1 2.67 -> 2.83 img/s (+6%), c=4 +2.7%, c=8/16 +1%.
- No regression risk for other models: nothing else imports this module. Merge optional; it can stay on perf/ovn-img.

## Verdict (integrator 22:40): reviewed OK, not merged into perf/overnight-shared; stays on perf/ovn-img

- Review of the shared hunk (`mstar/model/components/diffusion/qwen3/encoder.py`): `padded=None` keeps the device-side check, so callers that don't pass it (z_image) behave as before. The rotary cache is keyed by (seq, device). That is bounded: captions pad to multiples of 32 up to 512, so at most 16 entries. The values are the same tensors as before. Passing a host-known flag through step metadata is the pattern AGENTS.md invariant 7 asks for, and I found no invariant, S1-S4 or CPU-overhead concern.
- Not merged, because its only importers are flux2_klein and z_image, both owned by img, and the proposer marked the merge optional. A cross-model suite run would measure nothing it can change, and the regression box has only one GPU. img's own A/B covers it: klein bit-identical, c=1 +6%. Please re-check z_image on your branch, since it imports the same module without passing `padded`.
- If a non-img model starts importing this encoder, re-propose and it goes through the suite.
