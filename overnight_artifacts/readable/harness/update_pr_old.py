"""Assemble PR_BODY.md + TABLE.md + LOG.md into the draft PR's body; create the PR on first run (--create).
The PR lives on the user's fork (dghosef/mstar), base branch ovn-base-55b0215a3 = upstream main at the
night's base, so the diff is exactly the overnight changes."""
import pathlib
import re
import subprocess
import sys

D = pathlib.Path.home() / "runs/overnight"
body = (D / "PR_BODY.md").read_text()
body = re.sub(r"<!-- TABLE START -->.*<!-- TABLE END -->",
              lambda _: "<!-- TABLE START -->\n" + (D / "TABLE.md").read_text() + "\n<!-- TABLE END -->", body, flags=re.S)
body = re.sub(r"<!-- LOG START -->.*<!-- LOG END -->",
              lambda _: "<!-- LOG START -->\n" + (D / "LOG.md").read_text() + "\n<!-- LOG END -->", body, flags=re.S)
body = re.sub(r"<!-- SUMMARY START -->.*<!-- SUMMARY END -->",
              lambda _: "<!-- SUMMARY START -->\n" + (D / "SUMMARY_TABLE.md").read_text() + "\n<!-- SUMMARY END -->", body, flags=re.S)
body = re.sub(r"<!-- LATENCY START -->.*<!-- LATENCY END -->",
              lambda _: "<!-- LATENCY START -->\n" + (D / "LATENCY_TABLE.md").read_text() + "\n<!-- LATENCY END -->", body, flags=re.S)
body = re.sub(r"<!-- COVERAGE START -->.*<!-- COVERAGE END -->",
              lambda _: "<!-- COVERAGE START -->\n" + (D / "COVERAGE.md").read_text() + "\n<!-- COVERAGE END -->", body, flags=re.S)
body = re.sub(r"<!-- STATUS START -->.*<!-- STATUS END -->",
              lambda _: "<!-- STATUS START -->\n" + (D / "STATUS.md").read_text() + "\n<!-- STATUS END -->", body, flags=re.S)
import time
body = body.replace("## Gains by model", "_Last updated: " + time.strftime("%H:%M %Z, %b %d") + "._\n\n## Gains by model", 1)
out = D / "pr_body_rendered.md"
out.write_text(body)
num_f = D / "pr_number.txt"
R = ["--repo", "dghosef/mstar"]
if num_f.exists():
    # REST, not `gh pr edit`: the latter fails on the Projects (classic) GraphQL deprecation
    subprocess.run(["gh", "api", "-X", "PATCH", f"repos/dghosef/mstar/pulls/{num_f.read_text().strip()}",
                    "-F", f"body=@{out}", "--silent"], check=True)
    print("updated PR", num_f.read_text().strip())
elif "--create" in sys.argv:
    r = subprocess.run(["gh", "pr", "create", *R, "--draft", "--base", "main", "--head", "perf/overnight-all",
                        "--title", "[DO NOT MERGE] Overnight perf push 2026-10-21: tracking board for all model perf changes",
                        "--body-file", str(out)], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    m = re.search(r"/pull/(\d+)", r.stdout)
    if m:
        num_f.write_text(m.group(1))
