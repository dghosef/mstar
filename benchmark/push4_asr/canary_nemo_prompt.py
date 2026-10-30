"""The decoder prompt NeMo's Canary-1B-v2 transcribe() builds for an English transcription (canary2 prompt format,
the model's prompt_defaults), as token ids and pieces, to compare with mstar's config.prompt_ids.

  nemo/bin/python canary_nemo_prompt.py
"""
from nemo.collections.asr.models import ASRModel

m = ASRModel.from_pretrained("nvidia/canary-1b-v2", map_location="cpu").eval()
fmt = m.prompt
for slots in ({"source_lang": "en", "target_lang": "en"}, {"source_lang": "en", "target_lang": "en", "pnc": "yes"}):
    turns = [{"role": "user", "slots": {**{k: v for d in m.cfg.prompt_defaults if d["role"] == "user"
                                           for k, v in d["slots"].items()}, **slots}}]
    enc = fmt.encode_dialog(turns)
    ids = [int(i) for i in enc["context_ids"]]
    print(slots, "->", ids, m.tokenizer.ids_to_tokens(ids))
print("mstar ->", [7, 4, 16, 64, 64, 5, 9, 11, 13])
