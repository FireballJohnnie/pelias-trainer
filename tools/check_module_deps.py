#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Checks the framework rules: allowed module dependencies (acyclic), includes, CMake DEPS, provenance lines.

Usage: check_module_deps.py <repo root>.  Exit 0 = all good, 1 = at least one problem (each is printed).
The allowed graph for the modules of master 5.2 is the one printed there; the modules marked NEW are proposals (src/README.md).
"""
import re
import sys
from pathlib import Path

ALLOWED = {
    "hal": [],
    "config": [],
    "tok": [],
    "io": [
        "hal"
    ],
    "plan": [
        "hal"
    ],
    "ingest": [
        "io",
        "config"
    ],
    "data": [
        "io",
        "tok"
    ],
    "model": [
        "io"
    ],
    "stream": [
        "hal"
    ],
    "kern": [
        "hal"
    ],
    "sched": [
        "kern",
        "model",
        "stream"
    ],
    "train": [
        "sched",
        "model",
        "data"
    ],
    "export": [
        "io",
        "model",
        "config"
    ],
    "verify": [
        "hal",
        "io"
    ],
    "project": [
        "io"
    ],
    "startup": [
        "hal",
        "plan",
        "io",
        "project"
    ],
    "intake": [
        "data",
        "tok"
    ],
    "help": [],
    "warn": [
        "data",
        "verify",
        "intake"
    ],
    "flow": [
        "plan",
        "startup",
        "project",
        "intake",
        "warn",
        "verify",
        "help"
    ],
    "serve": [
        "project"
    ],
    "ui": [],
    "cli": [
        "train",
        "export",
        "verify",
        "flow",
        "serve"
    ]
}
NEW = ["project", "startup", "intake", "help", "warn", "flow", "serve", "ui"]


def closure(mod, seen=None):
    seen = set() if seen is None else seen
    for d in ALLOWED[mod]:
        if d not in seen:
            seen.add(d)
            closure(d, seen)
    return seen


def find_cycle():
    state = {}

    def visit(m, path):
        state[m] = 1
        for d in ALLOWED[m]:
            if state.get(d) == 1:
                return path + [m, d]
            if d not in state:
                found = visit(d, path + [m])
                if found:
                    return found
        state[m] = 2
        return None

    for m in ALLOWED:
        if m not in state:
            found = visit(m, [])
            if found:
                return found
    return None


def main(root):
    root = Path(root)
    problems = []
    cyc = find_cycle()
    if cyc:
        problems.append("dependency cycle: " + " -> ".join(cyc))
    src = root / "src"
    dirs = sorted(p.name for p in src.iterdir() if p.is_dir())
    for d in dirs:
        if d not in ALLOWED:
            problems.append("src/%s is not in the allowed graph" % d)
    for m in ALLOWED:
        if m not in dirs:
            problems.append("module %s has no src/%s" % (m, m))
    inc = re.compile(r'^\s*#\s*include\s+"([^"/]+)/[^"]+"', re.M)
    for m in dirs:
        if m not in ALLOWED:
            continue
        ok = closure(m) | {m}
        for f in sorted((src / m).rglob("*")):
            if f.suffix not in (".h", ".cc", ".cu", ".hpp"):
                continue
            for target in inc.findall(f.read_text(encoding="utf-8")):
                if target in ("pelias",):
                    continue
                if target in ALLOWED and target not in ok:
                    problems.append("%s includes module %s, which %s may not use" % (f.relative_to(root), target, m))
        cm = (src / m / "CMakeLists.txt").read_text(encoding="utf-8")
        mm = re.search(r"^# pelias-deps:(.*)$", cm, re.M)
        declared = set(mm.group(1).split()) if mm else None
        if declared is None:
            problems.append("src/%s/CMakeLists.txt has no '# pelias-deps:' line" % m)
        elif declared != set(ALLOWED[m]):
            problems.append("src/%s/CMakeLists.txt DEPS %s differ from the allowed graph %s" % (m, sorted(declared), sorted(ALLOWED[m])))
    for base in ("src", "include", "tests"):
        for f in sorted((root / base).rglob("*")):
            if f.suffix in (".h", ".cc", ".cu", ".hpp", ".in") and f.is_file():
                first = f.read_text(encoding="utf-8").split("\n", 1)[0]
                if not first.startswith("// provenance:"):
                    problems.append("%s: first line is not a provenance line" % f.relative_to(root))
    for p in problems:
        print("PROBLEM:", p)
    print("checked %d modules (%d new); %d problem(s)" % (len(dirs), len(NEW), len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
