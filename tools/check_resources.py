#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Checks the sheet templates (headers, EXAMPLE markers, source labels, code that parses) and the help files (entry format).

Usage: check_resources.py <repo root>.  Exit 0 = all good, 1 = at least one problem.
"""
import ast
import csv
import re
import sys
from pathlib import Path

HEADERS = {
    "persona.csv": ["key", "value", "status"],
    "parameters.csv": ["id", "name", "tier", "description", "answer_kind", "answer", "language", "depends_on", "status"],
    "clients.csv": ["client_name", "system_prompt_style", "system_prompt_text", "roles_used", "tool_schema", "returns_reasoning", "source", "status"],
}
SOURCES = {"CAPTURED", "PUBLISHED", "APPROXIMATION", "UNVERIFIED"}
STATUSES = {"EXAMPLE", "DRAFT", "READY"}


def check_sheets(root, problems):
    for name, header in HEADERS.items():
        path = root / "resources" / "sheets" / name
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if rows[0] != header:
            problems.append("%s: header is %s, expected %s" % (name, rows[0], header))
            continue
        col = {h: i for i, h in enumerate(header)}
        data = rows[1:]
        if not 3 <= len(data) <= 5:
            problems.append("%s: %d example rows (3 to 5 expected)" % (name, len(data)))
        for n, r in enumerate(data, 2):
            if len(r) != len(header):
                problems.append("%s row %d: %d fields" % (name, n, len(r)))
                continue
            if r[col["status"]] not in STATUSES:
                problems.append("%s row %d: unknown status %r" % (name, n, r[col["status"]]))
            if r[col["status"]] != "EXAMPLE":
                problems.append("%s row %d: the shipped templates may only hold EXAMPLE rows" % (name, n))
            if name == "clients.csv" and r[col["source"]] not in SOURCES:
                problems.append("%s row %d: source %r is not one of %s" % (name, n, r[col["source"]], sorted(SOURCES)))
            if name == "parameters.csv":
                if not r[col["id"]].isdigit():
                    problems.append("%s row %d: id is not a number" % (name, n))
                if r[col["answer_kind"]] not in ("code", "text", "none"):
                    problems.append("%s row %d: answer_kind %r" % (name, n, r[col["answer_kind"]]))
                if r[col["answer_kind"]] == "code" and r[col["language"]] == "python":
                    try:
                        ast.parse(r[col["answer"]])
                    except SyntaxError as exc:
                        problems.append("%s row %d: the python answer does not parse (%s)" % (name, n, exc.msg))


def check_help(root, problems):
    base = root / "resources" / "help"
    files = sorted(p for p in base.glob("*.txt") if p.name != "README.txt")
    if not files:
        problems.append("resources/help has no entry files")
    for f in files:
        keys = set()
        entry, fields = None, {}

        def close():
            if entry is None:
                return
            for need in ("short", "plain", "detail", "see"):
                if need not in fields:
                    problems.append("%s [%s]: missing %s" % (f.name, entry, need))
            if "short" in fields and ("\n" in fields["short"] or not fields["short"].strip()):
                problems.append("%s [%s]: short must be one non-empty line" % (f.name, entry))

        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            m = re.match(r"^\[([a-z0-9_]+)\]\s*$", line)
            if m:
                close()
                entry, fields = m.group(1), {}
                if entry in keys:
                    problems.append("%s: duplicate key %s" % (f.name, entry))
                keys.add(entry)
                continue
            m = re.match(r"^(short|plain|detail|see):\s*(.*)$", line)
            if m and entry:
                fields[m.group(1)] = m.group(2).strip()
                last = m.group(1)
            elif line.startswith("  ") and entry and fields:
                fields[last] += " " + line.strip()
            else:
                problems.append("%s: unexpected line %r" % (f.name, line[:60]))
        close()


def main(root):
    root = Path(root)
    problems = []
    check_sheets(root, problems)
    check_help(root, problems)
    for p in problems:
        print("PROBLEM:", p)
    print("resources checked; %d problem(s)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
