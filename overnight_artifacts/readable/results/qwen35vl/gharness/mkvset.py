import json, os
from PIL import Image, ImageDraw
out = "/root/guest-qwen35vl/vset"; os.makedirs(out, exist_ok=True)
items = []
for i, (w, h) in enumerate([(512, 384), (640, 480), (384, 512), (1536, 1152)]):
    im = Image.new("RGB", (w, h), (30 * i % 255, 120, 200)); d = ImageDraw.Draw(im)
    for k in range(10): d.ellipse([k * w // 12, k * h // 14, k * w // 12 + w // 6, k * h // 14 + h // 6], fill=(255, 40 * k % 255, 60))
    d.text((20, 20), f"Image number {i}", fill=(0, 0, 0))
    p = f"{out}/img{i}.png"; im.save(p)
    items.append({"id": f"img{i}", "kind": "image", "prompt": "Describe this image.", "image": p})
for v in ("t4s_360p", "t8s_360p", "t16s_360p", "t8s_720p"):
    for j, q in enumerate(("Describe this video.", "What changes over time in this video?")):
        items.append({"id": f"{v}_{j}", "kind": v, "prompt": q, "video": f"/root/guest-qwen35vl/vids/{v}.mp4"})
for i, t in enumerate(["Explain the difference between a list and a tuple in Python.", "Write a short poem about the ocean at night."]):
    items.append({"id": f"text{i}", "kind": "text", "prompt": t, "image": None})
json.dump(items, open(f"{out}/manifest.json", "w"), indent=1); print(len(items))
