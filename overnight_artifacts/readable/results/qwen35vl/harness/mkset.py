"""Build the image/text prompt set for qcheck.py (food101 validation via streaming)."""
import json
import sys
from pathlib import Path

from datasets import load_dataset

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("ethz/food101", split="validation", streaming=True)
prompts = [
    "What food dish is shown in this image?",
    "Describe the food item pictured here in detail.",
    "Identify the cuisine and dish visible in this photo.",
    "Generate a caption for this food image.",
]
items = []
imgs = []
for i, row in enumerate(ds):
    if i >= 20:
        break
    imgs.append(row["image"].convert("RGB"))
for i, im in enumerate(imgs[:12]):
    p = out / f"small_{i}.jpg"
    im.save(p, quality=95)
    items.append({"id": f"small_{i}", "kind": "small", "prompt": prompts[i % 4], "image": str(p),
                  "size": im.size})
for i, im in enumerate(imgs[12:16]):
    big = im.resize((1536, 1152))
    p = out / f"big_{i}.jpg"
    big.save(p, quality=95)
    items.append({"id": f"big_{i}", "kind": "big", "prompt": prompts[i % 4], "image": str(p),
                  "size": big.size})
for i, im in enumerate(imgs[16:18]):
    huge = im.resize((2048, 2048))
    p = out / f"huge_{i}.jpg"
    huge.save(p, quality=95)
    items.append({"id": f"huge_{i}", "kind": "huge", "prompt": prompts[i % 4], "image": str(p),
                  "size": huge.size})
texts = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean at night.",
    "What are three tips for improving sleep quality?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How does a refrigerator keep food cold?",
    "Give me a recipe for a simple tomato soup.",
    "What is the capital of Australia, and why was it chosen?",
    "Describe how photosynthesis works.",
]
for i, t in enumerate(texts):
    items.append({"id": f"text_{i}", "kind": "text", "prompt": t, "image": None})
(out / "manifest.json").write_text(json.dumps(items, indent=1))
print(len(items), "items")
