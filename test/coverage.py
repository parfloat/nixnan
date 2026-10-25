#!/usr/bin/env python3
"""Which instrumented opcodes do the tests exercise?

Disassembles every built test (cuobjdump -sass), collects the opcodes, and matches them the way
nixnan does (std::regex_match of the full opcode, e.g. "FADD.FTZ", against each key of
../src/instructions.json).  Writes results/<tag>/coverage.txt and prints it.
Exit status 1 if any table entry has no test, so it can gate CI.
"""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R

table = json.load(open(os.path.join(HERE, "..", "src", "instructions.json")))
plat = R.detect_platform()
OP = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9_.]*)")

seen = {}      # opcode -> set(tests)
for t in R.discover(HERE):
    exe = os.path.join(HERE, t)
    if not os.path.isfile(exe):
        continue
    out = subprocess.run(["cuobjdump", "-sass", exe], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = OP.search(line)
        if m:
            seen.setdefault(m.group(1), set()).add(t)

lines, untested = [], []
for key, v in table.items():
    tests = set()
    for op, ts in seen.items():
        if re.fullmatch(key, op):
            tests |= ts
    n = len(tests)
    desc = v.get("description", "")
    if n:
        lines.append("COVERED    %-52s %3d tests  e.g. %s" % (key, n, sorted(tests)[0]))
    else:
        lines.append("UNTESTED   %-52s %s" % (key, desc))
        untested.append(key)

# FP-looking opcodes the tests emit but nixnan's table does not instrument
FP_PREFIX = ("F", "H", "D", "MUFU", "I2F", "F2")
matched = lambda op: any(re.fullmatch(k, op) for k in table)
extra = sorted(op for op in seen if op.startswith(FP_PREFIX) and not matched(op))
lines.append("")
lines.append("FP-related opcodes present in the tests but NOT in instructions.json (not checked by nixnan):")
for op in extra:
    lines.append("           %-30s in %d test(s)" % (op, len(seen[op])))

text = "\n".join(lines) + "\n"
d = os.path.join(HERE, "results", plat["tag"])
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "coverage.txt"), "w").write(text)
print(text)
print("%d of %d table entries have no test on this platform" % (len(untested), len(table)))
sys.exit(1 if untested else 0)
