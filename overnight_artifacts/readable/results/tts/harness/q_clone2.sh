cd /root/work/tharness
until grep -q "QUEUE DONE" /root/work/q_ko8.out; do sleep 10; done
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf MSTAR_SHM_ARENA=1 USER=root PATH=/root/work/mstar/.venv/bin:$PATH
R=/workspace/results/clone2; mkdir -p $R
REF=/workspace/results/ko_base_r2/c1_s26_wav/0003.wav
REFTXT="$(sed -n 3p s26.txt)"
run() { # label tree mps config
  local label=$1 tree=$2 mps=$3 cfg=$4; local gpus=0
  export SERVER_STATE_DIR=/root/work/servers_8100
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  if [ "$mps" = 1 ]; then bash $tree/.claude/skills/benchmarking/scripts/server.sh mps-start 0; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_root_gpu0/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_root_gpu0/log; gpus=0; fi
  (cd $tree && PYTHONPATH=$tree bash .claude/skills/benchmarking/scripts/server.sh start $label 8100 --timeout 1800 --log $R/$label.server.log -- $P -m mstar.cli.main serve omnivoice --gpus $gpus --port 8100 $cfg)
  $P clone_smoke.py --port 8100 --ref $REF --ref-text "$REFTXT" --out $R/$label > $R/$label.txt 2>&1
  (cd $tree && bash .claude/skills/benchmarking/scripts/server.sh stop $label)
  if [ "$mps" = 1 ]; then sleep 3; bash $tree/.claude/skills/benchmarking/scripts/server.sh mps-stop 0; fi
}
run cl_base /root/work/base 0 ""
run cl_head /root/work/head3 0 ""
run cl_headsplit /root/work/head3 1 "--config configs/omnivoice_split.yaml"
export CUDA_VISIBLE_DEVICES=0
for t in cl_head cl_headsplit; do
$P quality.py --device cuda:0 --sentences $R/cl_base_sentences.txt --base $R/cl_base --test $R/$t --out $R/${t}_q.json > $R/${t}_q.txt 2>&1
done
echo QUEUE DONE
