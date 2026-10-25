#!/usr/bin/env python3
"""Run hello_nixnan under nixnan with PRINT_ILL_INSTR=1 and collect what it found.

  python3 collect_exceptions.py [path/to/nixnan.so]      (default ../../nixnan.so)

Writes, next to this script:
  hello_nixnan.report.txt   the end-of-run nixnan report (what a plain LD_PRELOAD run prints)
  hello_nixnan.detail.log   full stderr with one line per exception and the instruction that saw it
  exceptions.md             the same detail lines grouped by kernel / instruction / type
"""
import os, re, subprocess, sys, collections
here = os.path.dirname(os.path.abspath(__file__))
lib = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "nixnan.so"))
exe = os.path.join(here, "hello_nixnan")
env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
env.update(LD_PRELOAD=lib, PRINT_ILL_INSTR="1", ENABLE_FUN_DETAIL="1")
r = subprocess.run([exe], capture_output=True, text=True, env=env, cwd=here)
open(os.path.join(here, "hello_nixnan.detail.log"), "w").write(r.stderr)
i = r.stderr.find("#nixnan: ------------ nixnan Report")
open(os.path.join(here, "hello_nixnan.report.txt"), "w").write(r.stderr[i:])

pat = re.compile(r"error \[([^\]]+)\] detected in operand (\d+) of instruction (.*?) ; in function (\w+) of type (\w+)")
groups = collections.OrderedDict()
for line in r.stderr.splitlines():
    m = pat.search(line)
    if m:
        kinds, op, ins, fn, ty = m.groups()
        groups.setdefault((fn, ins.strip(), ty), []).append((kinds, op))

out = ["# hello_nixnan: exceptions nixnan found\n",
       "Collected by `collect_exceptions.py` (PRINT_ILL_INSTR=1). `operand 0` is the destination, "
       "`1..n` are the sources; `type` is the operand type nixnan decoded. An `STG` row is the "
       "memory path: a NaN being stored to global memory.\n"]
cur = None
for (fn, ins, ty), evs in groups.items():
    if fn != cur:
        out += ["", "## kernel `%s`" % fn, "", "| instruction | type | exceptions (operand) |", "|---|---|---|"]
        cur = fn
    what = ", ".join("%s (op %s)" % e for e in evs)
    out.append("| `%s` | %s | %s |" % (ins, ty, what))
out.append("")
open(os.path.join(here, "exceptions.md"), "w").write("\n".join(out))
print("%d instruction/exception groups; wrote exceptions.md, hello_nixnan.detail.log, hello_nixnan.report.txt" % len(groups))
