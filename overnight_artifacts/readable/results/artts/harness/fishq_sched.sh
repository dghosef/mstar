#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_w13.sh" >/dev/null; do sleep 15; done
rsync -a /root/work/t_head4/ /root/work/t_fish10/
python3 - <<'PY'
p="/root/work/t_fish10/configs/fish_s2_pro.yaml"; s=open(p).read()
s=s.replace("model_kwargs:\n  codec_exact_fp32: false", "model_kwargs:\n  codec_exact_fp32: false\n  chunk_schedule: [1, 3, 8, 16]")
open(p,"w").write(s)
PY
tail -3 /root/work/t_fish10/configs/fish_s2_pro.yaml
python ab.py --arms head=/root/work/t_head4,sched=/root/work/t_fish10 --rounds 2 --out /workspace/results/fish_sched -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32 --save-audio 16
