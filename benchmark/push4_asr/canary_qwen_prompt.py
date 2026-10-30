"""Canary-Qwen's decoder prompt for one transcription turn: NeMo SALM's prompt formatter (nemo side) vs mstar's
CanaryQwenModel.prompt_ids with one audio frame (mstar side), as token ids, for a side-by-side check.

  nemo/bin/python canary_qwen_prompt.py nemo
  PYTHONPATH=<tree> venv/bin/python canary_qwen_prompt.py mstar
"""
import sys

if sys.argv[1] == "nemo":
    from nemo.collections.common.prompts import PromptFormatter
    from nemo.collections.speechlm2.models import SALM

    m = SALM.from_pretrained("nvidia/canary-qwen-2.5b").eval()
    # as SALM.generate formats a prompt
    formatter = PromptFormatter.resolve(m.cfg.prompt_format)(m.tokenizer)
    turns = [{"role": "user", "content": f"Transcribe the following: {m.audio_locator_tag}"}]
    ids = [int(i) for i in formatter.encode_dialog(turns=turns)["input_ids"]]
    print("nemo ", ids)
    print("nemo text:", repr(m.tokenizer.ids_to_text(ids)))
else:
    from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel

    m = CanaryQwenModel("nvidia/canary-qwen-2.5b")
    ids = m.prompt_ids(1)
    print("mstar", ids)
    print("mstar text:", repr(m.tokenizer.decode(ids)))
