#!/bin/bash
# Coordinator check: new NOTES lines since last check, merge log, proposals, GPU use per box.
cd ~/runs/overnight
N=$(wc -l < NOTES.md); L=$(cat .last_notes 2>/dev/null || echo 0)
echo "== $(date +%H:%M)  disk: $(df -h /System/Volumes/Data | tail -1 | awk '{print $4" free"}')"
echo "== new NOTES ($L -> $N)"; tail -n +$((L+1)) NOTES.md; echo $N > .last_notes
echo "== LOG"; tail -5 shared/LOG.md
echo "== proposals"; ls -t shared/proposals/ | head -12
echo "== GPU util (12 s avg util%/memGB)"
for h in qwen35 qwen35vl omni artts video; do {
  r=$(timeout 35 ssh -F ssh_config -o ConnectTimeout=10 ovn-$h "for i in 1 2 3 4 5 6; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done | awk -F', ' '{u[\$1]+=\$2; m[\$1]=\$3; n[\$1]++} END {for (g in u) printf \"gpu%s %d%%/%dG  \", g, u[g]/n[g], m[g]/1024}'" 2>/dev/null)
  echo "  $h: ${r:-UNREACHABLE}"
  echo "$(date +%H:%M),$h,${r:-UNREACHABLE}" >> ~/runs/overnight/util_log.csv
  }
done
