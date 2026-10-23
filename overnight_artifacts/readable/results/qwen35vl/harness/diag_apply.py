"""Insert (or remove with --undo) the OVNSLOW slow-forward log into <tree>/mstar/engine/engine.py."""
import sys
p = sys.argv[1] + "/mstar/engine/engine.py"
s = open(p).read()
old = '''                phase_record("engine.forward", perf_counter() - t0)
            if nvtx:'''
new = '''                phase_record("engine.forward", perf_counter() - t0)
                _dt = perf_counter() - t0
                if _dt > 0.015:
                    logger.warning(
                        "OVNSLOW %.1fms node=%s walk=%s bs=%d tokens=%s lease=%s batched=%s",
                        _dt * 1e3, batch.node_name, batch.step_context.graph_walk,
                        len(request_ids), sum(getattr(i, "input_seq_len", 0) or 0 for i in batch.inputs),
                        lease is not None, running_batched,
                    )
            if nvtx:'''
if "--undo" in sys.argv:
    s = s.replace(new, old)
elif "OVNSLOW" not in s:
    assert s.count(old) == 1
    s = s.replace(old, new)
open(p, "w").write(s)
print("ok", "OVNSLOW" in s)
