cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ov5.out; do sleep 10; done
rm -rf /workspace/results/cb_base_g1_r1
for r in 1 2; do
$P bench.py --tree /root/work/mstar --label cb_base_g0_r$r --model chatterbox --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/exp4 --label cb_t3stage_g0_r$r --model chatterbox --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
done
$P bench.py --tree /root/work/exp4 --label cb_t3stage_splitmps_r1 --model chatterbox --config /root/work/cfg/chatterbox_split.yaml --mps --gpus 0 --port 8100 --points 32:s156:2
$P bench.py --asr --tree /root/work/exp3 --label hg_merge_phase --model higgs_audio --gpus 0 --port 8100 --phase 100 --points 32:200:_,1:50:_,32:200:_
$P bench.py --asr --tree /root/work/exp --label hg_new_phase --model higgs_audio --gpus 0 --port 8100 --phase 100 --points 32:200:_,1:50:_,32:200:_
echo QUEUE DONE
