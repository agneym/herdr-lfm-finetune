"""Herdr operations exposed as Needle 2 tools.

Each @needle.tool function wraps one `herdr` CLI operation.  Decorating with
@needle.tool makes the function carry a `_needle_tool` schema describing its
name, docstring (description) and parameter types -- exactly what the
fine-tuning dataset embeds as `tools` and what the runtime grammar is compiled
from.  Keeping the two identical is the whole game: the model is trained to emit
calls inside a grammar derived from these schemas.

Planner vs. executor
--------------------
By default the functions do NOT run the command; they return a descriptor
dict `{"name","command","arguments"}` that a caller (Hermes) can inspect and
execute itself.  Set NEEDLE_HERDR_EXECUTE=1 to make them shell out to the
installed `herdr` binary (and return the parsed JSON/result) -- used for the
optional full-loop self-execution mode.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Literal, Optional

import needle
from needle.agent.tools import build_schema

EXECUTE = os.environ.get("NEEDLE_HERDR_EXECUTE") == "1"


def _cli(*argv: str, env: Optional[dict] = None) -> object:
    """Run `herdr <argv...>`.

    Returns the parsed JSON body when the command prints JSON (herdr prints JSON
    for deterministic automation), otherwise the raw stdout text.  With
    EXECUTE=False this is never called.
    """
    try:
        proc = subprocess.run(
            ["herdr", *argv],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, **(env or {})},
        )
    except FileNotFoundError:
        return {"error": "herdr binary not found in PATH"}
    out = (proc.stdout or "").strip()
    if not out:
        out = (proc.stderr or "").strip()
    if out.startswith("{") or out.startswith("["):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    return {"exit": proc.returncode, "output": out}


def _desc(name: str, command: str, arguments: dict) -> dict:
    """Descriptor returned when NOT executing.  Hermes reads this to run the op."""
    return {"name": name, "command": command, "arguments": arguments}


# --------------------------------------------------------------------------- #
# Status / workspace / tab / session / worktree / integration
# --------------------------------------------------------------------------- #
@needle.tool
def herdr_status() -> dict:
    """Return the current Herdr client and server status (version, channel, health)."""
    command = "herdr status"
    _ = _cli("status") if EXECUTE else None
    return _desc("herdr_status", command, {})


@needle.tool
def workspace_list() -> dict:
    """List all Herdr workspaces with their ids, tabs, panes and labels."""
    command = "herdr workspace list"
    _ = _cli("workspace", "list") if EXECUTE else None
    return _desc("workspace_list", command, {})


@needle.tool
def workspace_create(
    cwd: Optional[str] = None,
    label: Optional[str] = None,
    focus: Optional[Literal["focus", "no-focus"]] = None,
) -> dict:
    """Create a new Herdr workspace with its initial tab and root pane.

    cwd: directory to open in the root pane. label: display name. focus: whether
    to move the TUI focus to the new workspace.
    """
    argv = ["workspace", "create"]
    if cwd:
        argv += ["--cwd", cwd]
    if label:
        argv += ["--label", label]
    if focus:
        argv += ["--" + focus]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("workspace_create", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"cwd": cwd, "label": label, "focus": focus})


@needle.tool
def workspace_get(workspace_id: str) -> dict:
    """Show a single workspace by its id (e.g. w4)."""
    command = f"herdr workspace get {workspace_id}"
    _ = _cli("workspace", "get", workspace_id) if EXECUTE else None
    return _desc("workspace_get", command, {"workspace_id": workspace_id})


@needle.tool
def tab_list(workspace: Optional[str] = None) -> dict:
    """List the tabs of a workspace. workspace: an optional workspace id; omit for all."""
    argv = ["tab", "list"]
    if workspace:
        argv += ["--workspace", workspace]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("tab_list", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"workspace": workspace})


@needle.tool
def tab_create(
    workspace: Optional[str] = None,
    cwd: Optional[str] = None,
    label: Optional[str] = None,
    focus: Optional[Literal["focus", "no-focus"]] = None,
) -> dict:
    """Open a new tab. workspace: optional workspace id (defaults to current).
    cwd: directory for the tab's root pane. label: display name. focus: whether to
    move the TUI focus to the new tab.
    """
    argv = ["tab", "create"]
    if workspace:
        argv += ["--workspace", workspace]
    if cwd:
        argv += ["--cwd", cwd]
    if label:
        argv += ["--label", label]
    if focus:
        argv += ["--" + focus]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("tab_create", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"workspace": workspace, "cwd": cwd, "label": label, "focus": focus})


@needle.tool
def session_list() -> dict:
    """List all named persistent Herdr sessions."""
    command = "herdr session list"
    _ = _cli("session", "list") if EXECUTE else None
    return _desc("session_list", command, {})


@needle.tool
def worktree_list(workspace: Optional[str] = None, cwd: Optional[str] = None) -> dict:
    """List git worktree-based workspaces. Pass workspace id or cwd to filter."""
    argv = ["worktree", "list"]
    if workspace:
        argv += ["--workspace", workspace]
    if cwd:
        argv += ["--cwd", cwd]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("worktree_list", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"workspace": workspace, "cwd": cwd})


@needle.tool
def worktree_create(
    workspace: Optional[str] = None,
    cwd: Optional[str] = None,
    branch: Optional[str] = None,
    base: Optional[str] = None,
    path: Optional[str] = None,
    label: Optional[str] = None,
    focus: Optional[Literal["focus", "no-focus"]] = None,
) -> dict:
    """Create a git worktree, open it as a workspace, group it with the parent repo workspace.
    workspace/cwd: identify the parent repo workspace. branch: branch name (created from base/HEAD
    if it does not exist). base: ref to branch from. path: checkout path (else under the configured
    worktrees directory). label: display name. focus: whether to focus it.
    """
    argv = ["worktree", "create"]
    if workspace:
        argv += ["--workspace", workspace]
    if cwd:
        argv += ["--cwd", cwd]
    if branch:
        argv += ["--branch", branch]
    if base:
        argv += ["--base", base]
    if path:
        argv += ["--path", path]
    if label:
        argv += ["--label", label]
    if focus:
        argv += ["--" + focus]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("worktree_create", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"workspace": workspace, "cwd": cwd, "branch": branch,
                  "base": base, "path": path, "label": label, "focus": focus})


@needle.tool
def integration_install(agent: Literal["pi", "omp", "claude", "codex", "copilot", "devin", "droid",
                                      "kimi", "opencode", "kilo", "hermes", "qodercli", "qwen",
                                      "cursor", "mastracode", "antigravity-cli", "grok"]) -> dict:
    """Install the built-in Herdr integration for an agent so Herdr recognizes it inside panes.
    agent: the agent kind whose integration to install.
    """
    command = f"herdr integration install {agent}"
    _ = _cli("integration", "install", agent) if EXECUTE else None
    return _desc("integration_install", command, {"agent": agent})


# --------------------------------------------------------------------------- #
# Pane primitives
# --------------------------------------------------------------------------- #
@needle.tool
def pane_list(workspace: Optional[str] = None) -> dict:
    """List the panes of a workspace (defaults to the caller's workspace).
    workspace: an optional workspace id."""
    argv = ["pane", "list"]
    if workspace:
        argv += ["--workspace", workspace]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("pane_list", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"workspace": workspace})


@needle.tool
def pane_current() -> dict:
    """Show the calling pane: its id, workspace, tab, cwd and recognized agent."""
    command = "herdr pane current --current"
    _ = _cli("pane", "current", "--current") if EXECUTE else None
    return _desc("pane_current", command, {})


@needle.tool
def pane_layout(pane: str) -> dict:
    """Show the layout (geometry) of a pane so you can decide a split direction.
    pane: the pane id (e.g. w4:p1).
    """
    command = f"herdr pane layout --pane {pane}"
    _ = _cli("pane", "layout", "--pane", pane) if EXECUTE else None
    return _desc("pane_layout", command, {"pane": pane})


@needle.tool
def pane_split(
    pane: Optional[str] = None,
    current: bool = False,
    direction: Literal["right", "down"] = "right",
    ratio: Optional[float] = None,
    cwd: Optional[str] = None,
    focus: Optional[Literal["focus", "no-focus"]] = None,
) -> dict:
    """Split a pane into two. pane: the pane id to split (omit to split the caller's pane).
    current: target the calling pane explicitly. direction: split to the right or down.
    ratio: the fraction of width/height for the new pane. cwd: working directory for the new
    pane (defaults to the caller's cwd). focus: whether to move focus to the new pane.
    """
    argv = ["pane", "split"]
    if current:
        argv += ["--current"]
    elif pane:
        argv += ["--pane", pane]
    argv += ["--direction", direction]
    if ratio is not None:
        argv += ["--ratio", str(ratio)]
    if cwd:
        argv += ["--cwd", cwd]
    if focus:
        argv += ["--" + focus]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("pane_split", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"pane": pane, "current": current, "direction": direction,
                  "ratio": ratio, "cwd": cwd, "focus": focus})


@needle.tool
def pane_run(pane: str, command: str) -> dict:
    """Run a shell command in a pane (sends the text and Enter atomically).
    pane: the pane id to run in. command: the shell command text.
    """
    cmd = f"herdr pane run {pane} {command}"
    _ = _cli("pane", "run", pane, command) if EXECUTE else None
    return _desc("pane_run", cmd, {"pane": pane, "command": command})


@needle.tool
def pane_read(
    pane: str,
    source: Literal["visible", "recent", "recent-unwrapped", "detection"] = "recent",
    lines: Optional[int] = None,
    format: Literal["text", "ansi"] = "text",
) -> dict:
    """Read a pane's terminal output. pane: the pane id. source: which snapshot (visible, recent,
    recent-unwrapped, detection). lines: max lines to read. format: text or ansi.
    """
    argv = ["pane", "read", pane]
    argv += ["--source", source]
    if lines is not None:
        argv += ["--lines", str(lines)]
    argv += ["--format", format]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("pane_read", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"pane": pane, "source": source, "lines": lines, "format": format})


@needle.tool
def pane_wait(
    pane: str,
    match: Optional[str] = None,
    regex: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    source: Literal["visible", "recent", "recent-unwrapped"] = "recent",
    lines: Optional[int] = None,
) -> dict:
    """Wait until a pane's output matches text (substring) or a regex. pane: the pane id.
    match: literal substring to search for. regex: Rust regular expression (use EITHER match or
    regex, not both). timeout_ms: fail after this many milliseconds (omit to wait forever).
    source: which snapshot to search. lines: restrict the searched snapshot to N lines.
    """
    argv = ["pane", "wait-output", pane]
    if match is not None:
        argv += ["--match", match]
    if regex is not None:
        argv += ["--regex", regex]
    if timeout_ms is not None:
        argv += ["--timeout", str(timeout_ms)]
    argv += ["--source", source]
    if lines is not None:
        argv += ["--lines", str(lines)]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("pane_wait", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"pane": pane, "match": match, "regex": regex,
                  "timeout_ms": timeout_ms, "source": source, "lines": lines})


@needle.tool
def pane_send_keys(pane: str, keys: list) -> dict:
    """Send key presses to a pane. pane: the pane id. keys: list of key names (e.g. "esc", "Enter",
    "Ctrl-c"). Use "esc" as the canonical Escape key name.
    """
    argv = ["pane", "send-keys", pane] + [str(k) for k in keys]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("pane_send_keys", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"pane": pane, "keys": keys})


@needle.tool
def pane_rename(pane: str, label: str) -> dict:
    """Rename a pane. pane: the pane id. label: the new name."""
    command = f"herdr pane rename {pane} {label}"
    _ = _cli("pane", "rename", pane, label) if EXECUTE else None
    return _desc("pane_rename", command, {"pane": pane, "label": label})


@needle.tool
def pane_close(pane: str) -> dict:
    """Close a pane. Only close panes you created unless the user explicitly asked."""
    command = f"herdr pane close {pane}"
    _ = _cli("pane", "close", pane) if EXECUTE else None
    return _desc("pane_close", command, {"pane": pane})


# --------------------------------------------------------------------------- #
# Agent primitives
# --------------------------------------------------------------------------- #
_AGENT_KINDS = ("pi", "claude", "codex", "gemini", "cursor", "devin", "agy", "cline",
                "omp", "mastracode", "opencode", "copilot", "kimi", "kiro", "droid",
                "amp", "grok", "hermes", "kilo", "qodercli", "qwen", "maki")


@needle.tool
def agent_list() -> dict:
    """List the recognized agents in Herdr with their lifecycle state."""
    command = "herdr agent list"
    _ = _cli("agent", "list") if EXECUTE else None
    return _desc("agent_list", command, {})


@needle.tool
def agent_start(
    name: str,
    kind: Literal["pi", "claude", "codex", "gemini", "cursor", "devin", "agy", "cline",
                  "omp", "mastracode", "opencode", "copilot", "kimi", "kiro", "droid",
                  "amp", "grok", "hermes", "kilo", "qodercli", "qwen", "maki"],
    pane: str,
    timeout_ms: Optional[int] = None,
    args: Optional[list] = None,
) -> dict:
    """Start a supported interactive coding agent in an existing shell pane.
    name: a unique live agent name matching [a-z][a-z0-9_-]{0,31}. kind: the agent kind.
    pane: the pane id at an interactive shell prompt. timeout_ms: wait for readiness
    (default 30000, max 300000). args: optional native agent arguments passed after --
    """
    argv = ["agent", "start", name, "--kind", kind, "--pane", pane]
    if timeout_ms is not None:
        argv += ["--timeout", str(timeout_ms)]
    _ = _cli(*argv) if EXECUTE else None
    if args and not EXECUTE:
        # keep the descriptor faithful without side effects
        argv += ["--"] + [str(a) for a in args]
    return _desc("agent_start", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"name": name, "kind": kind, "pane": pane,
                  "timeout_ms": timeout_ms, "args": args})


@needle.tool
def agent_get(target: str) -> dict:
    """Show a recognized agent's state. target: the unique live agent name or the pane id."""
    command = f"herdr agent get {target}"
    _ = _cli("agent", "get", target) if EXECUTE else None
    return _desc("agent_get", command, {"target": target})


@needle.tool
def agent_read(
    target: str,
    source: Literal["visible", "recent", "recent-unwrapped", "detection"] = "recent",
    lines: Optional[int] = None,
    format: Literal["text", "ansi"] = "text",
) -> dict:
    """Read a recognized agent's terminal output. target: the unique live agent name or pane id.
    source: which snapshot. lines: max lines. format: text or ansi.
    """
    argv = ["agent", "read", target]
    argv += ["--source", source]
    if lines is not None:
        argv += ["--lines", str(lines)]
    argv += ["--format", format]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("agent_read", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"target": target, "source": source, "lines": lines, "format": format})


@needle.tool
def agent_wait(
    target: str,
    until: Optional[list] = None,
    timeout_ms: Optional[int] = None,
) -> dict:
    """Wait until an agent reaches one of the requested lifecycle states.
    target: the unique live agent name or pane id. until: a list of states to match
    (idle, working, blocked, done, unknown); omit to match idle, done or blocked.
    timeout_ms: fail after this many milliseconds (omit to wait forever).
    """
    argv = ["agent", "wait", target]
    if until:
        for state in until:
            argv += ["--until", str(state)]
    if timeout_ms is not None:
        argv += ["--timeout", str(timeout_ms)]
    _ = _cli(*argv) if EXECUTE else None
    return _desc("agent_wait", "herdr " + " ".join(shlex.quote(a) for a in argv),
                 {"target": target, "until": until, "timeout_ms": timeout_ms})


# --------------------------------------------------------------------------- #
# Runtime wiring
# --------------------------------------------------------------------------- #
# Order matters only for readability; Needle handles any order.
_TOOLS = [
    herdr_status, workspace_list, workspace_create, workspace_get,
    tab_list, tab_create, session_list, worktree_list, worktree_create,
    integration_install, pane_list, pane_current, pane_layout, pane_split,
    pane_run, pane_read, pane_wait, pane_send_keys, pane_rename, pane_close,
    agent_list, agent_start, agent_get, agent_read, agent_wait,
]

SCHEMAS = [build_schema(fn) for fn in _TOOLS]


def get_tools():
    """Return the decorated callables for needle.Needle(tools=[...])."""
    return list(_TOOLS)


if __name__ == "__main__":
    # Print the tool catalogue as JSON (for generate-data --tools / inspection).
    print(json.dumps(SCHEMAS, indent=2))
