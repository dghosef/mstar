"""Render the draft tracking PR's body: PR_MAIN.md (the per-model table and summary) + the running
log (LOG.md) collapsed at the end, then PATCH it via REST (`gh pr edit` breaks on Projects-classic).
The old multi-section renderer is update_pr_old.py."""
import pathlib
import subprocess
import time

D = pathlib.Path.home() / "runs/overnight"
main = (D / "PR_MAIN.md").read_text()
log = (D / "LOG.md").read_text()
body = (main.rstrip() + f"\n\n_Last updated {time.strftime('%H:%M %Z, %b %d')}._\n\n"
        "<details><summary>Running log</summary>\n\n" + log.strip() + "\n\n</details>\n\n"
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")
out = D / "pr_body_rendered.md"
out.write_text(body)
num = (D / "pr_number.txt").read_text().strip()
subprocess.run(["gh", "api", "-X", "PATCH", f"repos/dghosef/mstar/pulls/{num}", "-F", f"body=@{out}", "--silent"], check=True)
print("updated PR", num, f"({len(body.splitlines())} lines)")
