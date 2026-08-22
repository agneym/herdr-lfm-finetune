#!/usr/bin/env python3
"""Live-validate data.jsonl against a real herdr server.

Creates a scratch workspace, replays every executable call pattern from the
dataset via the herdr CLI, checks agent flags against the real binaries,
validates regexes, and reports pass/fail/skip per case. Cleans up after.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERDR = shutil.which("herdr")
DATA = os.path.join(os.path.dirname(__file__), "data.jsonl")
results = []  # (row_idx, query, call, status, detail)


def hr(*argv, timeout=30):
    return subprocess.run([HERDR, *argv], capture_output=True, text=True,
                          timeout=timeout)


def jout(proc):
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {}


def record(i, q, call, status, detail=""):
    results.append((i, q, call, status, detail))


BIN = {"codex": "codex", "claude": "claude", "opencode": "opencode",
       "hermes": "hermes"}

rows = [json.loads(l) for l in open(DATA)]

# ---------------------------------------------------------------- setup ----
tmp = tempfile.mkdtemp(prefix="dsval-")
subprocess.run(["git", "init", "-q", "-b", "main", tmp], check=True)
# seed one commit so refs like 'main' resolve like a real checkout
open(os.path.join(tmp, "README.md"), "w").write("seed\n")
subprocess.run(["git", "-C", tmp, "add", "."], check=True)
subprocess.run(["git", "-C", tmp, "-c", "user.email=v@x", "-c",
                "user.name=v", "commit", "-qm", "seed"], check=True)
ws = jout(hr("workspace", "create", "--label", "ds-val", "--cwd", tmp))
if not ws.get("result"):
    print("FATAL: could not create scratch workspace:", ws)
    sys.exit(1)
W = ws["result"]["workspace"]["workspace_id"]      # e.g. w6
P1 = ws["result"]["root_pane"]["pane_id"]
panes = [P1]
print(f"scratch workspace {W} root pane {P1}")

next_pane = 2


def split(direction="right", ratio=None):
    """Split P1 to get a fresh idle shell pane; return its id."""
    global next_pane
    argv = ["pane", "split", "--pane", panes[0], "--direction", direction,
            "--no-focus"]
    if ratio:
        argv += ["--ratio", str(ratio)]
    r = jout(hr(*argv))
    pid = r.get("result", {}).get("pane", {}).get("pane_id")
    if not pid:
        raise RuntimeError(f"split failed: {r}")
    panes.append(pid)
    next_pane += 1
    # let the shell come up
    hr("pane", "wait-output", pid, "--regex", r"[\$#]\s*$", "--timeout", "10000")
    return pid


# --------------------------------------------------------------- replay ----
for i, row in enumerate(rows):
    q = row["query"]
    if not row["answers"]:
        continue
    for ans in row["answers"]:
        name, a = ans["name"], ans["arguments"]
        try:
            if name == "pane_split":
                pid = split(a.get("direction", "right"), a.get("ratio"))
                record(i, q, f"{name}", "PASS", f"new pane {pid}")
            elif name == "pane_run":
                cmd = a["command"]
                # only syntax-check; don't execute arbitrary builds
                p = subprocess.run(["bash", "-n", "-c", cmd],
                                   capture_output=True, text=True)
                status = "PASS" if p.returncode == 0 else "FAIL"
                record(i, q, f"{name}({cmd!r})", status, p.stderr.strip())
            elif name == "pane_wait":
                if "regex" in a:
                    try:
                        re.compile(a["regex"])
                        record(i, q, f"{name}.regex={a['regex']!r}", "PASS")
                    except re.error as e:
                        record(i, q, f"{name}.regex={a['regex']!r}", "FAIL", str(e))
                else:
                    record(i, q, name, "SKIP", "literal match, no live wait")
            elif name == "pane_read":
                src = a.get("source", "recent")
                p = hr("pane", "read", panes[0], "--source", src,
                       *(["--lines", str(a["lines"])] if "lines" in a else []),
                       *(["--format", a["format"]] if "format" in a else []))
                record(i, q, f"{name}(source={src})", 
                       "PASS" if p.returncode == 0 else "FAIL", p.stdout[:120])
            elif name == "pane_send_keys":
                keys = a["keys"]
                # key names must be single chars or known special keys
                ok = all(len(k) == 1 or k in ("Enter", "esc", "Tab",
                                              "Ctrl-c", "Ctrl-d") for k in keys)
                record(i, q, f"{name}{keys}", "PASS" if ok else "FAIL",
                       "" if ok else "unknown key name")
            elif name == "agent_start":
                kind = a["kind"]
                binname = BIN.get(kind)
                if not binname or not shutil.which(binname):
                    record(i, q, f"{name}(kind={kind})", "SKIP", "binary absent")
                    continue
                for flag in a.get("args", []):
                    if not flag.startswith("-"):
                        continue  # positional, fine
                    # binary must at least recognize the flag
                    p = subprocess.run([binname, flag, "--version"],
                                       capture_output=True, text=True, timeout=60)
                    bad = ("unexpected argument" in (p.stderr + p.stdout)
                           or "Unknown option" in (p.stderr + p.stdout)
                           or "unrecognized" in (p.stderr + p.stdout).lower())
                    record(i, q, f"{name} {kind} {flag}",
                           "FAIL" if bad else "PASS",
                           (p.stderr + p.stdout).strip().splitlines()[0] if bad else "")
            elif name in ("workspace_list", "tab_list", "pane_list", "session_list",
                          "worktree_list", "herdr_status", "pane_current",
                          "agent_list", "pane_layout"):
                cmap = {"herdr_status": ["status"],
                        "workspace_list": ["workspace", "list"],
                        "tab_list": ["tab", "list"],
                        "pane_list": ["pane", "list"],
                        "session_list": ["session", "list"],
                        "worktree_list": ["worktree", "list"],
                        "pane_current": ["pane", "current"],
                        "agent_list": ["agent", "list"],
                        "pane_layout": ["pane", "layout"]}
                p = hr(*cmap[name])
                record(i, q, name, "PASS" if p.returncode == 0 else "FAIL")
            elif name in ("workspace_create", "tab_create", "workspace_get",
                          "tab_create", "pane_rename", "pane_close", "pane_list"):
                record(i, q, name, "SKIP", "covered elsewhere / mutating")
            elif name == "worktree_create":
                base = a.get("base", "HEAD")
                p = subprocess.run(["git", "-C", tmp, "rev-parse",
                                    "--verify", base],
                                   capture_output=True, text=True)
                ok = p.returncode == 0 or base == "HEAD"
                record(i, q, f"worktree_create.base={base!r}",
                       "PASS" if ok else "FAIL",
                       "" if ok else "base ref does not exist in a fresh repo")
            elif name == "integration_install":
                valid = a["agent"] in ("hermes", "codex", "opencode", "claude")
                record(i, q, f"integration_install({a['agent']})",
                       "PASS" if valid else "FAIL")
            else:
                record(i, q, name, "SKIP", "not live-tested")
        except Exception as e:  # noqa: BLE001
            record(i, q, name, "FAIL", repr(e))

# --------------------------------------------------------------- summary ---
from collections import Counter
c = Counter(r[3] for r in results)
print("\n== summary ==", dict(c))
fails = [r for r in results if r[3] == "FAIL"]
for i, q, call, status, detail in fails:
    print(f"FAIL row {i}: {q[:70]}\n   {call}\n   {detail}")
with open(os.path.join(os.path.dirname(DATA), "validation_report.json"), "w") as f:
    json.dump([{"row": i, "query": q, "call": c_, "status": s, "detail": d}
               for i, q, c_, s, d in results], f, indent=1)

# --------------------------------------------------------------- cleanup ---
for pid in reversed(panes):
    hr("pane", "close", pid)
hr("workspace", "close", W)
shutil.rmtree(tmp, ignore_errors=True)
print("cleanup done")
