#!/usr/bin/env python3
"""Label PRs by the areas they touch (area/*, model/*) and their size (size/*).

Run by .github/workflows/pr-labels.yml with the default branch checked out;
reads only the PR's file list through the API, never its code. Locally:

    python3 .github/labels/label_prs.py --repo mstar-project/mstar --all --dry-run

Only labels under the managed prefixes are added or removed; anything else on
the PR (type/*, set by the review bot, or a human's labels) is left alone.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from urllib.parse import quote

MANAGED = ("area/", "model/", "size/")

# First matching prefix wins. mstar/model/<name>/ and configs/ are handled in
# area_of(), since their label depends on the model name.
AREAS = [
    ("mstar/api_server/", "area/api"),
    ("mstar/client/", "area/api"),
    ("mstar/cli/", "area/api"),
    ("rust/server/", "area/api"),
    ("mstar/conductor/", "area/conductor"),
    ("mstar/graph/", "area/conductor"),
    ("mstar/worker/", "area/worker"),
    ("mstar/communication/", "area/worker"),
    ("mstar/streaming/", "area/worker"),
    ("mstar/engine/", "area/engine"),
    ("mstar/distributed/", "area/engine"),
    ("mstar/platforms/", "area/engine"),
    ("mstar/utils/", "area/engine"),
    ("mstar/model/components/", "area/model-shared"),
    ("mstar/model/loader/", "area/model-shared"),
    ("rust/", "area/rust"),
    ("benchmark/", "area/perf"),
    ("perf_testing/", "area/perf"),
    ("mstar/profile/", "area/perf"),
    (".github/", "area/infra"),
    (".claude/", "area/infra"),
    ("packaging/", "area/infra"),
    ("scripts/", "area/infra"),
    ("pyproject.toml", "area/infra"),
    ("setup.py", "area/infra"),
    ("MANIFEST.in", "area/infra"),
    ("docs/", "area/docs"),
    ("test/", "area/tests"),
]
# Lockfiles say nothing about review effort.
IGNORED = ("uv.lock", "rust/Cargo.lock")

# A layer (area/*) needs this many changed lines, so a one-line touch doesn't
# label a PR with every layer it brushed against. Any change to a model's code
# or config labels that model.
MIN_LINES = 10
# A change across many models (an engine API every model implements, say) gets
# one model/many label rather than one per model.
MAX_MODELS = 3
SIZES = [(20, "size/XS"), (100, "size/S"), (700, "size/M"), (2000, "size/L"), (5000, "size/XL")]

COLORS = {"area/": "1d76db", "model/": "5319e7", "size/": "c5def5"}


def gh(*args, data=None):
    for attempt in range(3):
        out = subprocess.run(["gh", *args], input=data, capture_output=True, text=True, check=False)
        if not out.returncode:
            return out.stdout
        time.sleep(2 ** attempt)
    raise RuntimeError(f"gh {' '.join(args)}: {out.stderr.strip()}")


def gh_pages(path):
    """All items of a paginated list endpoint."""
    out = gh("api", "--paginate", "--slurp", path)
    return [item for page in json.loads(out) for item in page]


def area_of(path, models):
    if path in IGNORED:
        return None
    if path.startswith("mstar/model/"):
        parts = path.split("/")
        if len(parts) > 3 and parts[2] not in ("components", "loader"):
            return f"model/{parts[2]}"
    if path.startswith("configs/"):
        stem = path[len("configs/"):]
        # Longest name first, so qwen3_5_9b.yaml is qwen3_5, not some qwen3 model.
        for name in sorted(models, key=len, reverse=True):
            if stem == name or stem.startswith((name + "_", name + ".")):
                return f"model/{name}"
        return None
    for prefix, label in AREAS:
        if path.startswith(prefix):
            return label
    if "/" not in path and path.endswith(".md"):
        return "area/docs"
    return None


def labels_for(files, models):
    lines = defaultdict(int)
    for f in files:
        area = area_of(f["filename"], models)
        if area:
            lines[area] += f["additions"] + f["deletions"]
    size_lines = sum(n for a, n in lines.items() if a not in ("area/tests", "area/docs"))
    # Tests count toward no area, and docs only on a docs-only PR: nearly every
    # PR touches them. Test-only and docs-only PRs still get a label.
    code = {a: n for a, n in lines.items() if a not in ("area/tests", "area/docs")}
    if code:
        wanted = {a for a, n in code.items() if n >= MIN_LINES or a.startswith("model/")}
        if not wanted:
            wanted = {max(code, key=code.get)}
    elif "area/docs" in lines:
        wanted = {"area/docs"}
        size_lines = lines["area/docs"]
    elif "area/tests" in lines:
        wanted = {"area/tests"}
        size_lines = lines["area/tests"]
    else:
        wanted = set()
    models = {a for a in wanted if a.startswith("model/")}
    if len(models) > MAX_MODELS:
        wanted = (wanted - models) | {"model/many"}
    wanted.add(next((label for limit, label in SIZES if size_lines < limit), "size/XXL"))
    return wanted


def model_names(repo_root, files):
    names = set()
    model_dir = os.path.join(repo_root, "mstar", "model")
    if os.path.isdir(model_dir):
        names |= {d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))}
    # A new-model PR adds a folder the base doesn't have yet.
    for f in files:
        parts = f["filename"].split("/")
        if len(parts) > 3 and parts[:2] == ["mstar", "model"]:
            names.add(parts[2])
    return names - {"components", "loader", "__pycache__"}


def sync(repo, number, wanted, current, existing, dry_run):
    managed = {lb for lb in current if lb.startswith(MANAGED)}
    add, remove = sorted(wanted - managed), sorted(managed - wanted)
    print(f"#{number}: {' '.join(sorted(wanted))}"
          + (f"  (+{','.join(add)})" if add else "") + (f"  (-{','.join(remove)})" if remove else ""))
    if dry_run:
        return
    for label in add:
        if label not in existing:
            color = next(c for p, c in COLORS.items() if label.startswith(p))
            gh("api", "-X", "POST", f"repos/{repo}/labels", "-f", f"name={label}", "-f", f"color={color}")
            existing.add(label)
    if add:
        gh("api", "-X", "POST", f"repos/{repo}/issues/{number}/labels",
           "--input", "-", data=json.dumps({"labels": add}))
    for label in remove:
        gh("api", "-X", "DELETE", f"repos/{repo}/issues/{number}/labels/{quote(label, safe='')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("prs", nargs="*", type=int)
    ap.add_argument("--all", action="store_true", help="every open PR")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=".", help="checkout of the default branch, for the model list")
    a = ap.parse_args()

    if a.all:
        pulls = gh_pages(f"repos/{a.repo}/pulls?state=open&per_page=100")
        targets = [(p["number"], [lb["name"] for lb in p["labels"]]) for p in pulls]
    else:
        targets = [(n, [lb["name"] for lb in json.loads(gh("api", f"repos/{a.repo}/issues/{n}"))["labels"]])
                   for n in a.prs]
    existing = set() if a.dry_run else {lb["name"] for lb in gh_pages(f"repos/{a.repo}/labels?per_page=100")}
    failed = 0
    for number, current in targets:
        try:
            files = gh_pages(f"repos/{a.repo}/pulls/{number}/files?per_page=100")
            sync(a.repo, number, labels_for(files, model_names(a.root, files)),
                 set(current), existing, a.dry_run)
        except RuntimeError as e:
            print(f"#{number}: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
