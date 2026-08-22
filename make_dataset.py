"""Build the Herdr -> Needle 2 fine-tuning dataset.

Writes data.jsonl: one JSON object per line, each embedding the SAME tool
catalogue (`tools`) as the runtime, a representative environment `system`, a
natural-language `query`, a one-line `reasoning`, and the `answers` (exact
tool calls with arguments grounded in the query).

Run:  .venv/bin/python make_dataset.py [out.jsonl]
"""
import json
import os
import re
import sys

import herdr_tools as ht

# Representative environment facts. At runtime the same structure is passed via
# Needle(system=...); the model learns to read cwd / current-pane from here and
# to echo identifiers that the query names explicitly.
SYSTEM = (
    "HERDR_ENV=1\n"
    "workspace=w1\n"
    "tab=w1:t1\n"
    "pane=w1:p1\n"
    "cwd=/home/repo\n"
    "agent kind=hermes"
)

TOOLS = ht.SCHEMAS

# Each entry: (query, reasoning, answers).  Varied phrasing -> better grounding.
EXAMPLES = [
    # ---------------- status / discovery ----------------
    ("what's the herdr status?", "status tool; no args", [{
        "name": "herdr_status", "arguments": {}}]),
    ("is the herdr server healthy?", "status tool; no args", [{
        "name": "herdr_status", "arguments": {}}]),
    ("list all workspaces", "list workspaces; no args", [{
        "name": "workspace_list", "arguments": {}}]),
    ("show every workspace i have open", "list workspaces; no args", [{
        "name": "workspace_list", "arguments": {}}]),
    ("list the tabs in my current workspace", "tab_list with current workspace; no explicit id", [{
        "name": "tab_list", "arguments": {}}]),
    ("what tabs are in w1?", "tab_list; workspace id from query", [{
        "name": "tab_list", "arguments": {"workspace": "w1"}}]),
    ("show me the panes in workspace w2", "pane_list; workspace id from query", [{
        "name": "pane_list", "arguments": {"workspace": "w2"}}]),
    ("which pane am i in right now?", "pane_current; no args", [{
        "name": "pane_current", "arguments": {}}]),
    ("list the agents running right now", "agent_list; no args", [{
        "name": "agent_list", "arguments": {}}]),
    ("show all recognized coding agents", "agent_list; no args", [{
        "name": "agent_list", "arguments": {}}]),
    ("what named sessions exist?", "session_list; no args", [{
        "name": "session_list", "arguments": {}}]),
    ("list my persistent sessions", "session_list; no args", [{
        "name": "session_list", "arguments": {}}]),
    ("which worktrees are open?", "worktree_list; no filter", [{
        "name": "worktree_list", "arguments": {}}]),
    ("list worktrees for the repo at /home/repo", "worktree_list; cwd filter from query", [{
        "name": "worktree_list", "arguments": {"cwd": "/home/repo"}}]),
    ("what is workspace w3?", "workspace_get; id from query", [{
        "name": "workspace_get", "arguments": {"workspace_id": "w3"}}]),

    # ---------------- create / layout ----------------
    ("make a new workspace", "workspace_create; no args", [{
        "name": "workspace_create", "arguments": {}}]),
    ("open a new workspace at /home/repo/proj", "workspace_create; cwd from query", [{
        "name": "workspace_create", "arguments": {"cwd": "/home/repo/proj"}}]),
    ("create a workspace called frontend", "workspace_create; label from query", [{
        "name": "workspace_create", "arguments": {"label": "frontend"}}]),
    ("new tab please", "tab_create; no args", [{
        "name": "tab_create", "arguments": {}}]),
    ("open a tab for /home/repo/tests", "tab_create; cwd from query", [{
        "name": "tab_create", "arguments": {"cwd": "/home/repo/tests"}}]),
    ("make a tab named deploy in w1", "tab_create; label and workspace from query", [{
        "name": "tab_create", "arguments": {"workspace": "w1", "label": "deploy"}}]),

    # ---------------- pane layout ----------------
    ("show me the layout of pane w1:p2", "pane_layout; pane id from query", [{
        "name": "pane_layout", "arguments": {"pane": "w1:p2"}}]),
    ("split this pane to the right", "pane_split; current pane, direction right", [{
        "name": "pane_split", "arguments": {"current": True, "direction": "right"}}]),
    ("split my pane down", "pane_split; current pane, direction down", [{
        "name": "pane_split", "arguments": {"current": True, "direction": "down"}}]),
    ("split pane w1:p1 to the right and focus it", "pane_split; pane from query, right, focus", [{
        "name": "pane_split", "arguments": {"pane": "w1:p1", "direction": "right", "focus": "focus"}}]),
    ("split my pane to the right in /home/repo without stealing focus",
     "pane_split; current, right, cwd from query, no-focus", [{
        "name": "pane_split", "arguments": {"current": True, "direction": "right",
                                            "cwd": "/home/repo", "focus": "no-focus"}}]),
    ("split my pane down and give half the height", "pane_split; current, down, ratio from query", [{
        "name": "pane_split", "arguments": {"current": True, "direction": "down", "ratio": 0.5}}]),

    # ---------------- pane run / read / wait / keys / rename / close ----------------
    ("run `just test` in pane w1:p3", "pane_run; pane and command from query", [{
        "name": "pane_run", "arguments": {"pane": "w1:p3", "command": "just test"}}]),
    ("in w1:p1 run the test suite", "pane_run; pane and command from query", [{
        "name": "pane_run", "arguments": {"pane": "w1:p1", "command": "cargo test"}}]),
    ("run `npm run build` in my other pane", "pane_run; pane from query, command from query", [{
        "name": "pane_run", "arguments": {"pane": "w1:p2", "command": "npm run build"}}]),
    ("read the last 120 lines of w1:p1", "pane_read; pane and lines from query", [{
        "name": "pane_read", "arguments": {"pane": "w1:p1", "lines": 120}}]),
    ("show the recent output from pane w2:p1", "pane_read; pane from query; default recent", [{
        "name": "pane_read", "arguments": {"pane": "w2:p1"}}]),
    ("dump everything visible in w1:p3", "pane_read; pane from query; visible source", [{
        "name": "pane_read", "arguments": {"pane": "w1:p3", "source": "visible"}}]),
    ("wait for 'test result' in w1:p1 up to 2 minutes", "pane_wait; match, timeout from query", [{
        "name": "pane_wait", "arguments": {"pane": "w1:p1", "match": "test result",
                                           "timeout_ms": 120000}}]),
    ("wait until w1:p2 prints 'BUILD SUCCESS'", "pane_wait; match from query", [{
        "name": "pane_wait", "arguments": {"pane": "w1:p2", "match": "BUILD SUCCESS"}}]),
    ("send ctrl-c to pane w2:p1", "pane_send_keys; pane and key from query", [{
        "name": "pane_send_keys", "arguments": {"pane": "w2:p1", "keys": ["Ctrl-c"]}}]),
    ("send `q` to the pane w1:p4 (quit pager)", "pane_send_keys; pane and key from query", [{
        "name": "pane_send_keys", "arguments": {"pane": "w1:p4", "keys": ["q"]}}]),
    ("rename pane w1:p1 to builder", "pane_rename; pane and label from query", [{
        "name": "pane_rename", "arguments": {"pane": "w1:p1", "label": "builder"}}]),
    ("close pane w1:p4", "pane_close; pane from query", [{
        "name": "pane_close", "arguments": {"pane": "w1:p4"}}]),

    # ---------------- agents ----------------
    ("start a codex agent called reviewer in pane w1:p2",
     "agent_start; name, kind, pane from query", [{
        "name": "agent_start", "arguments": {"name": "reviewer", "kind": "codex", "pane": "w1:p2"}}]),
    ("launch claude agent 'coder' in w1:p1", "agent_start; name, kind, pane from query", [{
        "name": "agent_start", "arguments": {"name": "coder", "kind": "claude", "pane": "w1:p1"}}]),
    ("start an opencode agent called debug in w1:p3",
     "agent_start; name, kind, pane from query", [{
        "name": "agent_start", "arguments": {"name": "debug", "kind": "opencode", "pane": "w1:p3"}}]),
    ("start a hermes agent named triage in w2:p1",
     "agent_start; name, kind, pane from query", [{
        "name": "agent_start", "arguments": {"name": "triage", "kind": "hermes", "pane": "w2:p1"}}]),
    ("check agent reviewer", "agent_get; target from query", [{
        "name": "agent_get", "arguments": {"target": "reviewer"}}]),
    ("what's the state of w1:p2's agent?", "agent_get; target pane id from query", [{
        "name": "agent_get", "arguments": {"target": "w1:p2"}}]),
    ("read the last 120 lines of agent reviewer", "agent_read; target, lines from query", [{
        "name": "agent_read", "arguments": {"target": "reviewer", "lines": 120}}]),
    ("show recent output from the coder agent", "agent_read; target from query", [{
        "name": "agent_read", "arguments": {"target": "coder"}}]),
    ("wait until agent reviewer is done", "agent_wait; target and state from query", [{
        "name": "agent_wait", "arguments": {"target": "reviewer", "until": ["done"]}}]),
    ("wait for agent debug to be idle or working", "agent_wait; target and states from query", [{
        "name": "agent_wait", "arguments": {"target": "debug", "until": ["idle", "working"]}}]),

    # ---------------- worktree / integration ----------------
    ("create a worktree for branch feature/x", "worktree_create; branch from query", [{
        "name": "worktree_create", "arguments": {"branch": "feature/x"}}]),
    ("new worktree from main at /tmp/repo-x", "worktree_create; base and path from query", [{
        "name": "worktree_create", "arguments": {"base": "main", "path": "/tmp/repo-x"}}]),
    ("open a worktree on branch fix/y", "worktree_create; branch from query", [{
        "name": "worktree_create", "arguments": {"branch": "fix/y", "focus": "focus"}}]),
    ("install the hermes integration for herdr", "integration_install; agent from query", [{
        "name": "integration_install", "arguments": {"agent": "hermes"}}]),
    ("set up the codex integration", "integration_install; agent from query", [{
        "name": "integration_install", "arguments": {"agent": "codex"}}]),

    # ---------------- off-topic (answers: []) ----------------
    ("what is the capital of France?", "off-topic", []),
    ("write me a poem about the ocean", "off-topic", []),
    ("explain how transformers work", "off-topic", []),
    ("generate a fractal pattern", "off-topic", []),
    ("calculate 47 * 31 for me", "off-topic", []),
]

# A few multi-call coordination examples (split -> run -> wait).  These teach
# sequencing: each answer element is one call in order.
MULTI = [
    ("split my pane right and run the linter in the new pane, waiting for it to finish",
     "split then run then wait; pane from new split, cwd current, match from query", [
        {"name": "pane_split", "arguments": {"current": True, "direction": "right",
                                             "focus": "no-focus"}},
        {"name": "pane_run", "arguments": {"pane": "w1:p2", "command": "cargo fmt --check"}},
        # cargo fmt --check prints nothing on success, so match on the shell
        # prompt return rather than a literal output string.
        {"name": "pane_wait", "arguments": {"pane": "w1:p2", "regex": "\\$\\s*$",
                                            "timeout_ms": 60000}}]),
    ("start a codex reviewer in a fresh pane and wait for it to be ready",
     "split, start agent, wait; pane from new split", [
        {"name": "pane_split", "arguments": {"current": True, "direction": "right",
                                             "focus": "no-focus"}},
        {"name": "agent_start", "arguments": {"name": "reviewer", "kind": "codex", "pane": "w1:p2"}},
        {"name": "agent_wait", "arguments": {"target": "reviewer", "until": ["idle"]}}]),
]

OFF_RULE = ("Use the herdr tools only when the request is about controlling, "
            "inspecting or configuring a Herdr terminal workspace, pane, tab, "
            "workspace, worktree, session or coding agent. Off-topic requests "
            "return no tool call.")

# --------------------------------------------------------------------------- #
# Templated synthesis: deterministic, domain-correct phrasing variety.
# Each tuple fills a template to yield a grounded (query, reasoning, answers).
# --------------------------------------------------------------------------- #
CWD = "/home/repo"


def _queries(phrasings, glue=("please", "would you", "")):
    """Yield phrasing variants: each phrasing x a chosen glue prefix."""
    out = []
    for ph in phrasings:
        for g in glue:
            if g:
                out.append(f"{g}, {ph}")
            else:
                out.append(ph)
    return out


def synth():
    from itertools import product

    rows = []

    def add(*phr_answers):
        """phr_answers: alternating (query, answer_args) pairs sharing a reasoning."""
        pass

    # status / list / discovery -----------------------------------------------
    for ph in ["what's the herdr status?", "is the herdr server healthy?",
               "check herdr status", "how is herdr running?"]:
        rows.append((ph, "status; no args", [{"name": "herdr_status", "arguments": {}}]))
    for ph in ["list all workspaces", "show my workspaces", "what workspaces are open?",
               "list every workspace"]:
        rows.append((ph, "workspace_list; no args", [{"name": "workspace_list", "arguments": {}}]))
    for ph in ["list the agents", "show running agents", "what agents are active?"]:
        rows.append((ph, "agent_list; no args", [{"name": "agent_list", "arguments": {}}]))
    for ph in ["list sessions", "what named sessions exist?", "show my sessions"]:
        rows.append((ph, "session_list; no args", [{"name": "session_list", "arguments": {}}]))
    for ph in ["list the panes in w1", "show panes in workspace w1"]:
        rows.append((ph, "pane_list; workspace from query", [{"name": "pane_list", "arguments": {"workspace": "w1"}}]))
    for ph in ["list the tabs in w1", "show tabs of w1", "what tabs are in w1?"]:
        rows.append((ph, "tab_list; workspace from query", [{"name": "tab_list", "arguments": {"workspace": "w1"}}]))
    for ph in ["which pane am I in?", "show my current pane", "what's the current pane?"]:
        rows.append((ph, "pane_current; no args", [{"name": "pane_current", "arguments": {}}]))
    for ph in ["what is workspace w3?", "show me w3", "describe workspace w3"]:
        rows.append((ph, "workspace_get; id from query", [{"name": "workspace_get", "arguments": {"workspace_id": "w3"}}]))
    for ph in ["list worktrees", "what worktrees are open?", "show open worktrees"]:
        rows.append((ph, "worktree_list; no filter", [{"name": "worktree_list", "arguments": {}}]))
    for ph in ["worktrees for /home/repo", "list worktrees at /home/repo"]:
        rows.append((ph, "worktree_list; cwd from query", [{"name": "worktree_list", "arguments": {"cwd": "/home/repo"}}]))

    # create ------------------------------------------------------------------
    for ph in ["make a new workspace", "create a workspace", "new workspace"]:
        rows.append((ph, "workspace_create; no args", [{"name": "workspace_create", "arguments": {}}]))
    for ph in ["create a workspace at /home/repo/proj", "open a workspace in /home/repo/proj"]:
        rows.append((ph, "workspace_create; cwd from query", [{"name": "workspace_create", "arguments": {"cwd": "/home/repo/proj"}}]))
    for ph in ["make a workspace called api", "create a workspace named api"]:
        rows.append((ph, "workspace_create; label from query", [{"name": "workspace_create", "arguments": {"label": "api"}}]))
    for ph in ["new tab", "open a tab", "make a tab"]:
        rows.append((ph, "tab_create; no args", [{"name": "tab_create", "arguments": {}}]))
    for ph in ["open a tab at /home/repo/tests", "new tab in /home/repo/tests"]:
        rows.append((ph, "tab_create; cwd from query", [{"name": "tab_create", "arguments": {"cwd": "/home/repo/tests"}}]))
    for ph in ["tab named deploy", "create a tab labeled deploy"]:
        rows.append((ph, "tab_create; label from query", [{"name": "tab_create", "arguments": {"label": "deploy"}}]))

    # pane layout -------------------------------------------------------------
    for ph in ["show the layout of w1:p2", "what's the layout of pane w1:p2?"]:
        rows.append((ph, "pane_layout; pane id from query", [{"name": "pane_layout", "arguments": {"pane": "w1:p2"}}]))
    for ph, dr in [("split this pane to the right", "right"),
                   ("split my pane down", "down"),
                   ("split this pane right", "right"),
                   ("divide my pane on the right side", "right")]:
        rows.append((ph, f"pane_split; current pane, direction {dr}",
                     [{"name": "pane_split", "arguments": {"current": True, "direction": dr}}]))
    for ph, dr in [("split pane w1:p1 to the down", "down"),
                   ("split w1:p3 to the right", "right"),
                   ("split w1:p2 down", "down"),
                   ("split pane w1:p4 to the right", "right")]:
        pane = re.search(r"w\d+:p\d+", ph).group(0)
        rows.append((ph, f"pane_split; pane from query, direction {dr}",
                     [{"name": "pane_split", "arguments": {"pane": pane, "direction": dr}}]))
    for ph, dr, ratio in [("split my pane down giving a third of the height", "down", 0.33),
                          ("split my pane right at 30%", "right", 0.3),
                          ("split my pane down with the new pane taking two thirds", "down", 0.67)]:
        rows.append((ph, f"pane_split; current, direction {dr}, ratio from query",
                     [{"name": "pane_split", "arguments": {"current": True, "direction": dr, "ratio": ratio}}]))
    for ph, dr, cwd, nofocus in [
            ("split my pane right in /home/repo without focus", "right", "/home/repo", "no-focus"),
            ("split this pane down in /home/repo/proj, don't focus", "down", "/home/repo/proj", "no-focus")]:
        rows.append((ph, f"pane_split; current, direction {dr}, cwd from query, focus {nofocus}",
                     [{"name": "pane_split", "arguments": {"current": True, "direction": dr,
                                                           "cwd": cwd, "focus": nofocus}}]))

    # pane run / read / wait / keys / rename / close --------------------------
    for pane, cmd in [("w1:p3", "just test"), ("w1:p1", "cargo test"),
                      ("w1:p2", "npm run build"), ("w2:p1", "pytest -q"),
                      ("w1:p4", "go test ./..."), ("w2:p2", "bun run dev"),
                      ("w1:p1", "git status"), ("w2:p1", "cargo fmt --check")]:
        rows.append((f"run `{cmd}` in {pane}", f"pane_run; pane and command from query",
                     [{"name": "pane_run", "arguments": {"pane": pane, "command": cmd}}]))
        rows.append((f"in {pane} run {cmd}", f"pane_run; pane and command from query",
                     [{"name": "pane_run", "arguments": {"pane": pane, "command": cmd}}]))
    for pane, lines in [("w1:p1", 120), ("w1:p4", 50), ("w2:p1", 80), ("w1:p2", 200),
                        ("w2:p2", 150), ("w1:p3", 40)]:
        rows.append((f"read the last {lines} lines of {pane}", f"pane_read; pane and lines from query",
                     [{"name": "pane_read", "arguments": {"pane": pane, "lines": lines}}]))
    for pane, src in [("w2:p1", "recent"), ("w1:p3", "visible"), ("w1:p1", "recent-unwrapped"),
                      ("w2:p2", "recent"), ("w1:p4", "visible")]:
        rows.append((f"show the {src} output from {pane}", f"pane_read; pane and source from query",
                     [{"name": "pane_read", "arguments": {"pane": pane, "source": src}}]))
    for pane, src in [("w1:p1", "recent"), ("w1:p2", "visible")]:
        rows.append((f"read the {src} output from {pane} as ansi", f"pane_read; pane, source, ansi format",
                     [{"name": "pane_read", "arguments": {"pane": pane, "source": src, "format": "ansi"}}]))
    for pane in ["w1:p1", "w1:p3"]:
        rows.append((f"show the detection snapshot of {pane}", f"pane_read; pane and detection source",
                     [{"name": "pane_read", "arguments": {"pane": pane, "source": "detection"}}]))
    for pane, match, ms in [("w1:p1", "test result", 120000), ("w1:p2", "BUILD SUCCESS", 60000),
                            ("w2:p1", "all tests passed", 90000), ("w1:p4", "ok", 30000),
                            ("w2:p2", "listening on", 45000)]:
        rows.append((f"wait for '{match}' in {pane} up to {ms} ms", f"pane_wait; match and timeout from query",
                     [{"name": "pane_wait", "arguments": {"pane": pane, "match": match, "timeout_ms": ms}}]))
    for pane, match in [("w1:p2", "Build succeeded"), ("w1:p1", "error:"),
                        ("w2:p1", "No tests failed"), ("w1:p3", "launched")]:
        rows.append((f"wait until {pane} shows {match}", f"pane_wait; match from query",
                     [{"name": "pane_wait", "arguments": {"pane": pane, "match": match}}]))
    for pane, rx, ms in [("w1:p1", r"error:\s*\d+", 60000),
                         ("w1:p2", r"Build (succeeded|failed)", 90000),
                         ("w2:p1", r"test result: (ok|FAILED)", None)]:
        if ms:
            rows.append((f"wait until {pane} matches {rx} up to {ms} ms", f"pane_wait; regex, timeout from query",
                         [{"name": "pane_wait", "arguments": {"pane": pane, "regex": rx, "timeout_ms": ms}}]))
        else:
            rows.append((f"wait until {pane} matches {rx}", f"pane_wait; regex from query",
                         [{"name": "pane_wait", "arguments": {"pane": pane, "regex": rx}}]))
    for pane, key in [("w2:p1", "Ctrl-c"), ("w1:p4", "q"), ("w1:p1", "esc"), ("w2:p2", "Ctrl-c")]:
        rows.append((f"send {key} to {pane}", f"pane_send_keys; pane and key from query",
                     [{"name": "pane_send_keys", "arguments": {"pane": pane, "keys": [key]}}]))
    for pane, keys in [("w1:p1", ["Ctrl-c", "y"]), ("w2:p1", ["Enter"]), ("w1:p2", ["esc", "q"])]:
        rows.append((f"send {' then '.join(keys)} to {pane}", f"pane_send_keys; pane and keys from query",
                     [{"name": "pane_send_keys", "arguments": {"pane": pane, "keys": keys}}]))
    for pane, label in [("w1:p1", "builder"), ("w1:p2", "runner"), ("w2:p1", "test"),
                        ("w1:p4", "watch")]:
        rows.append((f"rename pane {pane} to {label}", f"pane_rename; pane and label from query",
                     [{"name": "pane_rename", "arguments": {"pane": pane, "label": label}}]))
    for pane in ["w1:p4", "w1:p3", "w2:p2"]:
        rows.append((f"close pane {pane}", f"pane_close; pane from query",
                     [{"name": "pane_close", "arguments": {"pane": pane}}]))

    # agents ------------------------------------------------------------------
    for name, kind, pane in [("reviewer", "codex", "w1:p2"), ("coder", "claude", "w1:p1"),
                             ("debug", "opencode", "w1:p3"), ("triage", "hermes", "w2:p1"),
                             ("test", "codex", "w1:p2")]:
        rows.append((f"start a {kind} agent called {name} in {pane}",
                     f"agent_start; name, kind, pane from query",
                     [{"name": "agent_start", "arguments": {"name": name, "kind": kind, "pane": pane}}]))
        rows.append((f"launch {kind} '{name}' in {pane}",
                     f"agent_start; name, kind, pane from query",
                     [{"name": "agent_start", "arguments": {"name": name, "kind": kind, "pane": pane}}]))
    for name, kind, pane in [("reviewer", "codex", "w1:p2")]:
        rows.append((f"start {name} ({kind}) in {pane}, wait up to 120s",
                     "agent_start; name, kind, pane, timeout from query",
                     [{"name": "agent_start", "arguments": {"name": name, "kind": kind, "pane": pane,
                                                            "timeout_ms": 120000}}]))
    for target in ["reviewer", "coder", "w1:p2"]:
        rows.append((f"check agent {target}", f"agent_get; target from query",
                     [{"name": "agent_get", "arguments": {"target": target}}]))
    for target, lines in [("reviewer", 120), ("coder", 60), ("w1:p2", 200)]:
        rows.append((f"read the last {lines} lines of agent {target}",
                     f"agent_read; target and lines from query",
                     [{"name": "agent_read", "arguments": {"target": target, "lines": lines}}]))
    for target, states in [("reviewer", ["done"]), ("debug", ["idle", "working"]),
                           ("coder", ["blocked", "done"])]:
        rows.append((f"wait until agent {target} is {' or '.join(states)}",
                     f"agent_wait; target and states from query",
                     [{"name": "agent_wait", "arguments": {"target": target, "until": states}}]))
    for target, states, ms in [("debug", ["idle", "working"], 120000),
                               ("coder", ["done"], 90000)]:
        rows.append((f"wait until agent {target} is {' or '.join(states)} up to {ms // 1000}s",
                     f"agent_wait; target, states, timeout from query",
                     [{"name": "agent_wait", "arguments": {"target": target, "until": states, "timeout_ms": ms}}]))
    rows.append(("start a codex agent called review in w1:p2 with --dangerously-bypass-approvals-and-sandbox",
                 "agent_start; name, kind, pane, args from query",
                 [{"name": "agent_start", "arguments": {"name": "review", "kind": "codex", "pane": "w1:p2",
                                                        "args": ["--dangerously-bypass-approvals-and-sandbox"]}}]))
    rows.append(("start a claude agent called review in w1:p2 with --dangerously-skip-permissions",
                 "agent_start; name, kind, pane, args from query",
                 [{"name": "agent_start", "arguments": {"name": "review", "kind": "claude", "pane": "w1:p2",
                                                        "args": ["--dangerously-skip-permissions"]}}]))
    rows.append(("launch claude 'doc' in w1:p3 with --print and --output-format text",
                 "agent_start; name, kind, pane, args from query",
                 [{"name": "agent_start", "arguments": {"name": "doc", "kind": "claude", "pane": "w1:p3",
                                                        "args": ["--print", "--output-format", "text"]}}]))

    # worktree / integration --------------------------------------------------
    for branch in ["feature/x", "fix/y", "feat/z"]:
        rows.append((f"create a worktree for branch {branch}",
                     f"worktree_create; branch from query",
                     [{"name": "worktree_create", "arguments": {"branch": branch}}]))
        # "open" moves focus, "create" does not (matches the hand-written seed).
        rows.append((f"open a worktree on branch {branch}",
                     f"worktree_create; branch and focus from query",
                     [{"name": "worktree_create", "arguments": {"branch": branch, "focus": "focus"}}]))
    for base, path in [("main", "/tmp/repo-x"), ("HEAD", "/tmp/repo-y")]:
        rows.append((f"new worktree from {base} at {path}",
                     f"worktree_create; base and path from query",
                     [{"name": "worktree_create", "arguments": {"base": base, "path": path}}]))
    for pane_id, branch in [("w1", "hotfix/x"), ("w2", "hotfix/y")]:
        rows.append((f"create a worktree for branch {branch} in the workspace {pane_id}",
                     f"worktree_create; workspace and branch from query",
                     [{"name": "worktree_create", "arguments": {"workspace": pane_id, "branch": branch}}]))
    rows.append(("create a worktree for branch fix/q at /home/repo",
                 "worktree_create; cwd and branch from query",
                 [{"name": "worktree_create", "arguments": {"cwd": "/home/repo", "branch": "fix/q"}}]))
    rows.append(("create a worktree from main labeled ft-main",
                 "worktree_create; base and label from query",
                 [{"name": "worktree_create", "arguments": {"base": "main", "label": "ft-main"}}]))
    for agent in ["hermes", "codex", "opencode"]:
        rows.append((f"install the {agent} integration",
                     f"integration_install; agent from query",
                     [{"name": "integration_install", "arguments": {"agent": agent}}]))

    # workspace / tab focus ---------------------------------------------------
    rows.append(("create a workspace at /home/repo/front and focus it",
                 "workspace_create; cwd and focus from query",
                 [{"name": "workspace_create", "arguments": {"cwd": "/home/repo/front", "focus": "focus"}}]))
    rows.append(("open a workspace at /home/repo/back without focus",
                 "workspace_create; cwd and focus from query",
                 [{"name": "workspace_create", "arguments": {"cwd": "/home/repo/back", "focus": "no-focus"}}]))
    rows.append(("open a tab at /home/repo/tests and focus it",
                 "tab_create; cwd and focus from query",
                 [{"name": "tab_create", "arguments": {"cwd": "/home/repo/tests", "focus": "focus"}}]))
    rows.append(("open a tab for /home/repo/docs without stealing focus",
                 "tab_create; cwd and focus from query",
                 [{"name": "tab_create", "arguments": {"cwd": "/home/repo/docs", "focus": "no-focus"}}]))

    # off-topic ---------------------------------------------------------------
    off = ["what is the capital of France?",
           "write me a limerick", "explain quantum computing",
           "translate hello to Spanish", "what's 84 * 12?",
           "tell me a joke", "write a python quickstart", "summarize the news",
           "order me a pizza", "what's the weather like?", "find me a recipe",
           "what time is it in Tokyo?", "draft an email to my boss",
           "what does this error stack trace mean?",
           "recommend me a book", "who won the 1998 world cup?",
           "how do I file my taxes?", "plan a two-week trip to Japan",
           "what is the meaning of life?", "compare rust and go",
           "help me name my cat", "what's the best pizza topping?",
           "write a haiku about autumn"]
    for q in off:
        rows.append((q, "off-topic", []))

    return rows


def main(out="data.jsonl"):
    rows = []
    seen = set()

    def push(query, reasoning, answers, tag):
        key = (query.strip().lower(), json.dumps(answers, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        rows.append({"query": query, "reasoning": reasoning, "answers": answers,
                     "tools": TOOLS, "system": SYSTEM})

    for query, reasoning, answers in EXAMPLES + MULTI:
        push(query, reasoning, answers, "hand")
    for query, reasoning, answers in synth():
        push(query, reasoning, answers, "synth")

    with open(out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    off = sum(1 for r in rows if not r["answers"])
    print(f"wrote {out}: {len(rows)} examples ({off} off-topic)")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.jsonl")
