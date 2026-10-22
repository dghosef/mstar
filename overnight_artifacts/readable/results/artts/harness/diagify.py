#!/usr/bin/env python3
"""Add a per-walk/batch-size gpu_exec span to a tree's worker (diagnostic only)."""
import sys
p = sys.argv[1] + "/mstar/worker/worker.py"
s = open(p).read()
s = s.replace("""            start, end = self._gpu_spans[0]
            if not end.query():""", """            start, end, walk = self._gpu_spans[0]
            if not end.query():""")
s = s.replace("""            self._phase_record("gpu_exec", start.elapsed_time(end) / 1000)""", """            self._phase_record("gpu_exec", start.elapsed_time(end) / 1000)
            self._phase_record("gpu_exec." + walk, start.elapsed_time(end) / 1000)""")
s = s.replace("self._gpu_spans.append((gpu_start, gpu_end))",
              "self._gpu_spans.append((gpu_start, gpu_end, batch.graph_walk + \".bs\" + str(len(node_batch.request_ids))))")
assert "gpu_exec.\" + walk" in s
open(p, "w").write(s)
