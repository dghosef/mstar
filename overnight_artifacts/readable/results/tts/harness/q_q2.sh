cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf CUDA_VISIBLE_DEVICES=0
until grep -q "QUEUE DONE" /root/work/q_vc.out; do sleep 10; done
R=/workspace/results; mkdir -p $R/quality
q() { $P quality.py --device cuda:0 --sentences $1 --base $R/$2 --test $R/$3 --out $R/quality/$4.json > $R/quality/$4.txt 2>&1; }
q ov150.txt ov5_graph_r2/c32_ov150_wav ov5_fix_r2/c32_ov150_wav ov_fix_vs_graph
q ov150.txt ov5_graph_r2/c32_ov150_wav ov5_fixsplit_r2/c32_ov150_wav ov_fixsplit_vs_graph
q s156.txt cb_base_g0_r1/c32_s156_wav cb_t3stage_g0_r1/c32_s156_wav cb_t3stage_vs_base
q s156.txt cb_novc_r1/c32_s156_wav cb_vc_r1/c32_s156_wav cb_vc_vs_novc
echo QUEUE DONE
