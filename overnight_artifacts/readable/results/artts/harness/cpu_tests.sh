#!/bin/bash
# CPU Core suite on my head and the shared base, for a failure-set comparison
export CUDA_VISIBLE_DEVICES= TORCHINDUCTOR_COMPILE_THREADS=8 MAX_JOBS=16
for T in t_head base2; do
  cd /root/work/$T
  PYTHONPATH=/root/work/$T timeout 3600 /root/work/mstar/.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/pt_$T test/modular test/distributed $(ls -d mstar/model/*/tests 2>/dev/null) > /workspace/results/cpu_core_$T.txt 2>&1
done
echo CPU TESTS DONE
