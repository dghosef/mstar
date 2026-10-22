"""Keep perf/overnight-all (the never-merge tracking branch behind the draft PR) equal to
perf/overnight-shared + every agent branch, as the agents' branches are NOW.

Each run builds a fresh merge on perf/overnight-all-build, then adds one commit on top of
perf/overnight-all whose tree is exactly that build, with both as parents. Pushes stay
fast-forward (force-push is not allowed) and nothing an agent has since dropped lingers.
On a conflict the agent's side wins; a branch that still won't merge is skipped and reported."""
import json
import subprocess
import sys

WT = "/Users/josephtan/mstar-ovn-all"
AGENTS = ["qwen35", "qwen35vl", "omni", "artts", "tts", "img", "video", "world", "coord"]


def git(*a, check=True):
    return subprocess.run(["git", "-C", WT, *a], capture_output=True, text=True, check=check)


git("checkout", "-q", "-B", "perf/overnight-all-build", "perf/overnight-shared")
status = {}
for a in AGENTS:
    ref = f"perf/ovn-{a}"
    sha = git("rev-parse", "--short", ref).stdout.strip()
    r = git("merge", "--no-ff", "--no-edit", "-m", f"overnight-all: merge {ref}", ref, check=False)
    if not r.returncode:
        status[a] = "merged " + sha
        continue
    files = git("diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
    git("merge", "--abort", check=False)
    r = git("merge", "--no-ff", "--no-edit", "-X", "theirs", "-m", f"overnight-all: merge {ref} (agent side wins)", ref, check=False)
    if r.returncode:
        git("merge", "--abort", check=False)
        status[a] = "SKIPPED, conflict: " + " ".join(files[:6])
    else:
        status[a] = f"merged {sha} (agent side on {' '.join(files[:6])})"

build = git("rev-parse", "HEAD").stdout.strip()
git("checkout", "-q", "perf/overnight-all")
if git("rev-parse", "HEAD^{tree}").stdout != git("rev-parse", f"{build}^{{tree}}").stdout:
    msg = "overnight-all: sync to the current agent branches\n\n" + "\n".join(f"{a}: {s}" for a, s in status.items())
    new = git("commit-tree", f"{build}^{{tree}}", "-p", "HEAD", "-p", build, "-m", msg).stdout.strip()
    git("reset", "-q", "--hard", new)
big = [ln for ln in git("diff", "--stat=200", "55b0215a3..HEAD").stdout.splitlines() if "Bin " in ln]
print(json.dumps({"head": git("rev-parse", "--short", "HEAD").stdout.strip(), "status": status, "binaries": big}, indent=1))
if "--push" in sys.argv:
    p = git("push", "-q", "dghosef", "perf/overnight-all", check=False)
    print("push rc", p.returncode, p.stderr[-300:])
