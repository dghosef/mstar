"""Check .github/review/session.py: the review's usage record and its /review continue handoff."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION = runpy.run_path(str(REPO_ROOT / ".github/review/session.py"))
load, usage, handoff, diff_paths = SESSION["load"], SESSION["usage"], SESSION["handoff"], SESSION["diff_paths"]
ROOT = "/home/runner/work/mstar/mstar"


def assistant(mid, *content, out=10, parent=None):
    return {"type": "assistant", "parent_tool_use_id": parent, "message": {
        "id": mid, "model": "claude-opus-5-5", "content": list(content),
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
                  "output_tokens": out}}}


def tool(tid, name, **inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def result(tid, error=None):
    block = {"type": "tool_result", "tool_use_id": tid, "content": error or "ok"}
    if error:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}}


def text(t):
    return {"type": "text", "text": t}


COMMENT = "mcp__github_inline_comment__create_inline_comment"


class TestUsage(unittest.TestCase):
    def test_one_response_over_several_lines_counts_once(self):
        msgs = [assistant("msg_a", text("x"), out=5), assistant("msg_a", tool("t1", "Read"), out=40),
                assistant("msg_b", text("y")),
                {"type": "result", "num_turns": 2, "total_cost_usd": 0.5, "subtype": "success"}]
        u = usage(msgs)
        self.assertEqual(u["api_calls"], 2, "each content block's line would count as an API call")
        self.assertEqual((u["cache_read_input_tokens"], u["output_tokens"]), (2000, 50))
        self.assertEqual((u["num_turns"], u["total_cost_usd"]), (2, 0.5))

    def test_synthetic_messages_are_not_api_calls(self):
        msg = assistant("msg_s", text("Credit balance is too low"))
        msg["message"]["model"] = "<synthetic>"
        self.assertEqual(usage([msg])["api_calls"], 0)

    def test_execution_file_and_transcript_together_count_once(self):
        msgs = [assistant("msg_a", text("x")), assistant("msg_b", text("y"))]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "exec.json").write_text(json.dumps(msgs))
            proj = Path(d, "projects", "-home-runner")
            proj.mkdir(parents=True)
            Path(proj, "sid.jsonl").write_text("\n".join(map(json.dumps, msgs)) + "\nnot json\n")
            u = usage(load([str(Path(d, "exec.json")), str(Path(d, "projects")), str(Path(d, "missing.json"))]))
        self.assertEqual(u["api_calls"], 2)
        self.assertIsNone(u["num_turns"], "a run without a result message has no turn count")


class TestHandoff(unittest.TestCase):
    def run_messages(self):
        return [
            assistant("m1", text("Reading the diff."), tool("r1", "Read", file_path=f"{ROOT}/pr-head/mstar/a.py")),
            assistant("m2", text("Progress:\n- P1 `mstar/a.py:10` drops a token, verified"),
                      tool("r2", "Read", file_path=f"{ROOT}/pr-head/mstar/b.py", offset=100, limit=50)),
            assistant("m3", tool("c1", COMMENT, path="mstar/a.py", line=10, body="**[P1]** drops a token\n\nmore")),
            result("c1"),
            assistant("m4", tool("c2", COMMENT, path="mstar/b.py", line=5, body="**[P2]** stale comment")),
            result("c2", error="line 5 is not in the diff"),
            assistant("m5", text("subagent text"), parent="task1"),
        ]

    def test_records_comments_reads_and_the_latest_progress_note(self):
        h = handoff(self.run_messages(), ROOT, ["mstar/a.py", "mstar/b.py", "mstar/c.py"])
        status = {c["path"]: c["status"] for c in h["comments"]}
        self.assertEqual(status, {"mstar/a.py": "posted", "mstar/b.py": "failed"})
        self.assertEqual(h["reads"], ["pr-head/mstar/a.py", "pr-head/mstar/b.py"])
        self.assertEqual(h["not_opened"], ["mstar/c.py"])
        self.assertTrue(h["note"].startswith("Progress:"))
        self.assertEqual(h["last_text"], h["note"], "a subagent's text would replace the reviewer's notes")
        md = SESSION["handoff_md"](h, "568", "abc", "light")
        self.assertIn("## Inline comments already posted\n\n- `mstar/a.py:10` **[P1]** drops a token", md)
        self.assertIn("line 5 is not in the diff", md)

    def test_a_continued_run_merges_onto_its_handoff(self):
        first = json.loads(json.dumps(handoff(self.run_messages(), ROOT, ["mstar/a.py", "mstar/c.py"])))
        second = handoff([assistant("n1", text("No note here"),
                                    tool("r3", "Read", file_path=f"{ROOT}/pr-head/mstar/c.py"))],
                         ROOT, ["mstar/a.py", "mstar/c.py"], first)
        self.assertEqual(second["runs"], 2)
        self.assertEqual(len(second["comments"]), 2, "the first run's inline comments would be forgotten")
        self.assertEqual(second["not_opened"], [])
        self.assertTrue(second["note"].startswith("Progress:"), "the last progress note would be lost")
        self.assertEqual(second["last_text"], "No note here")

    def test_diff_paths_skip_deleted_files(self):
        diff = "diff --git a/x b/x\n--- a/x\n+++ /dev/null\ndiff --git a/y b/y\n--- a/y\n+++ b/y\n"
        self.assertEqual(diff_paths(diff), ["y"])


if __name__ == "__main__":
    unittest.main()
