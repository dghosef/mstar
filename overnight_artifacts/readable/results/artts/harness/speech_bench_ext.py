#!/usr/bin/env python3
"""benchmark.tts_speech_bench with extra request fields read from files.

    python speech_bench_ext.py --extra-file ref_audio=/root/work/ref_b64.txt <tts_speech_bench args>
(no argv size limit on the field's value).
"""
import sys

from benchmark import tts_speech_bench

argv, extra = [], []
it = iter(sys.argv[1:])
for a in it:
    if a == "--extra-file":
        key, _, path = next(it).partition("=")
        extra += ["--extra", f"{key}={open(path).read().strip()}"]
    else:
        argv.append(a)
tts_speech_bench.main(argv + extra)
