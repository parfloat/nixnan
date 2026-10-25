#!/usr/bin/env python3
"""nixnan regression runner.

Runs each test executable under LD_PRELOAD=<nixnan.so>, compares the nixnan report with the
expected one, and collects every report under results/<platform-tag>/.

  run.py nixnan.so                      check all runnable tests (default mode)
  run.py nixnan.so --only fp32          only tests under fp32/  (or fp32/div_single, or a prefix)
  run.py nixnan.so --record             (re)write <test>.expect
  run.py nixnan.so --record-platform    (re)write <test>.<sm>.expect (e.g. div_single.sm86.expect)
  run.py --platform                     print the detected platform and exit
  run.py --list-runnable                print the tests that apply to this platform and exit

Outcomes: PASS, FAIL (report differs), ERROR (no report, crash, timeout, not built),
SKIP (platform does not satisfy the test's @min_sm/@max_sm), MISSING (no .expect).
Exit status is 1 if any test FAILed, ERRORed or is MISSING; skips alone exit 0.

Test annotations (comment lines in the .cu):  // @min_sm 80   // @max_sm 89   // @timeout 300
Expectation lookup: <test>.<sm>.expect (e.g. sqrt_single.sm86.expect) first, then <test>.expect.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "#nixnan: ------------ nixnan Report -----------"
DEFAULT_TIMEOUT = 120


# ---------------------------------------------------------------- platform
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def detect_platform():
    """GPU name, compute capability, driver, CUDA toolkit. NIXNAN_TEST_SM overrides the SM
    (used to exercise the skip logic, e.g. NIXNAN_TEST_SM=70)."""
    p = {"gpu": "unknown", "sm": None, "driver": "unknown", "nvcc": "unknown"}
    q = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
              "--format=csv,noheader"]).splitlines()
    if q:
        name, cc, drv = [x.strip() for x in q[0].split(",")]
        p["gpu"], p["driver"] = name, drv
        try:
            p["sm"] = int(cc.replace(".", ""))
        except ValueError:
            pass
    m = re.search(r"release (\d+\.\d+)", _run(["nvcc", "--version"]))
    if m:
        p["nvcc"] = m.group(1)
    if os.environ.get("NIXNAN_TEST_SM"):
        p["sm"] = int(os.environ["NIXNAN_TEST_SM"])
        p["gpu"] += " (SM overridden by NIXNAN_TEST_SM)"
    p["tag"] = "sm%s_cuda%s" % (p["sm"], p["nvcc"])
    return p


# ---------------------------------------------------------------- tests
def annotation(path, key, default=None):
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"\s*//\s*@%s\s+(\d+)" % key, line)
            if m:
                return int(m.group(1))
    return default


def discover(root):
    """Tests are exactly the */*.cu files (not every executable that happens to exist)."""
    tests = []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if os.path.isdir(full) and d != "results":
            for f in sorted(os.listdir(full)):
                if f.endswith(".cu"):
                    tests.append(os.path.join(d, f[:-3]))
    return tests


def applies(test, plat):
    """None if the test applies, else the reason it is skipped."""
    src = os.path.join(HERE, test + ".cu")
    lo, hi = annotation(src, "min_sm"), annotation(src, "max_sm")
    sm = plat["sm"]
    if sm is None:
        return None  # unknown platform: try it
    if lo is not None and sm < lo:
        return "needs sm_%d, have sm_%d" % (lo, sm)
    if hi is not None and sm > hi:
        return "needs sm <= %d, have sm_%d" % (hi, sm)
    return None


def expect_path(test, plat):
    cands = []
    if plat["sm"] is not None:
        cands.append(os.path.join(HERE, "%s.sm%d.expect" % (test, plat["sm"])))
    cands.append(os.path.join(HERE, test + ".expect"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def selected(tests, only):
    if not only:
        return tests
    return [t for t in tests if any(t == o.rstrip("/") or t.startswith(o) for o in only)]


# ---------------------------------------------------------------- one test
def run_test(test, env, timeout):
    exe = os.path.join(HERE, test)
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        return None, "executable not built (run: make %s)" % test
    try:
        r = subprocess.run([exe], capture_output=True, text=True, env=env,
                           timeout=timeout, cwd=HERE)
    except subprocess.TimeoutExpired:
        return None, "timeout after %ds" % timeout
    i = r.stderr.find(MARKER)
    if i < 0:
        return None, "no nixnan report in stderr (exit code %d)" % r.returncode
    rep = r.stderr[i:]
    if not rep.endswith("\n"):
        rep += "\n"
    return rep + "Exit Code: %d\n" % r.returncode, None


def nonzero(report):
    """[(section, kind, unique, repeats)] for every non-zero counter in a report."""
    out, sec = [], None
    for line in report.splitlines():
        m = re.match(r"#nixnan: --- (.*?) ---", line)
        if m:
            sec = re.sub(r"\s+", " ", m.group(1).strip())
            continue
        m = re.match(r"#nixnan: ([A-Za-z\- ]+?):\s+(\d+) \((\d+) repeats\)", line)
        if m and sec and int(m.group(2)) > 0:
            out.append((sec, m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lib", nargs="?", help="path to nixnan.so")
    ap.add_argument("--only", action="append", help="test, directory or prefix (repeatable)")
    ap.add_argument("--record", action="store_true", help="write <test>.expect")
    ap.add_argument("--record-platform", action="store_true", help="write <test>.<sm>.expect")
    ap.add_argument("--platform", action="store_true", help="print platform and exit")
    ap.add_argument("--list-runnable", action="store_true", help="print applicable tests and exit")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--keep-ld-library-path", action="store_true",
                    help="do not unset LD_LIBRARY_PATH for the tests")
    a = ap.parse_args()

    plat = detect_platform()
    tests = discover(HERE)
    if a.platform:
        print(json.dumps(plat, indent=2))
        return 0
    if a.list_runnable:
        print("\n".join(t for t in tests if applies(t, plat) is None))
        return 0
    if not a.lib:
        ap.error("path to nixnan.so required")
    lib = os.path.abspath(a.lib)
    if not (os.path.isfile(lib) and lib.endswith(".so")):
        sys.exit("error: %s is not a .so file" % lib)

    env = os.environ.copy()
    env["LD_PRELOAD"] = lib
    if not a.keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)

    print("Platform: %s, sm_%s, driver %s, nvcc %s  [%s]" %
          (plat["gpu"], plat["sm"], plat["driver"], plat["nvcc"], plat["tag"]))
    print("nixnan:   %s" % lib)
    mode = "record" if a.record else "record-platform" if a.record_platform else "check"
    print("Mode:     %s\n" % mode)

    outdir = os.path.join(a.results_dir, plat["tag"])
    os.makedirs(outdir, exist_ok=True)
    counts, rows, failed = {}, [], []

    chosen = selected(tests, a.only)
    if not chosen:
        sys.exit("error: no test matches %s" % a.only)
    for t in chosen:
        why = applies(t, plat)
        if why:
            res, detail = "SKIP", why
        else:
            timeout = annotation(os.path.join(HERE, t + ".cu"), "timeout", DEFAULT_TIMEOUT)
            rep, err = run_test(t, env, timeout)
            if rep is None:
                res, detail = "ERROR", err
            else:
                os.makedirs(os.path.join(outdir, os.path.dirname(t)), exist_ok=True)
                with open(os.path.join(outdir, t + ".report"), "w") as f:
                    f.write(rep)
                rows += [(t,) + x for x in nonzero(rep)]
                if a.record or a.record_platform:
                    suffix = ".sm%d.expect" % plat["sm"] if a.record_platform else ".expect"
                    dst = os.path.join(HERE, t + suffix)
                    with open(dst, "w") as f:
                        f.write(rep)
                    res, detail = "RECORDED", os.path.relpath(dst, HERE)
                else:
                    ep = expect_path(t, plat)
                    if ep is None:
                        res, detail = "MISSING", "no %s.expect" % t
                    elif open(ep).read() == rep:
                        res, detail = "PASS", ""
                    else:
                        res, detail = "FAIL", "report differs from %s" % os.path.relpath(ep, HERE)
        counts[res] = counts.get(res, 0) + 1
        if res in ("FAIL", "ERROR", "MISSING"):
            failed.append(t)
        print("%-8s %-28s %s" % (res, t, detail))

    # Collected exceptions: one row per non-zero counter (merged with earlier partial runs).
    tsv = os.path.join(outdir, "exceptions.tsv")
    old = {}
    if os.path.exists(tsv):
        for line in open(tsv).read().splitlines()[1:]:
            old.setdefault(line.split("\t")[0], []).append(line)
    for t in {r[0] for r in rows} | {t for t in selected(tests, a.only)}:
        old.pop(t, None)
    for r in rows:
        old.setdefault(r[0], []).append("%s\t%s\t%s\t%d\t%d" % r)
    with open(tsv, "w") as f:
        f.write("test\tsection\tkind\tunique\trepeats\n")
        for t in sorted(old):
            f.write("\n".join(old[t]) + "\n")
    with open(os.path.join(outdir, "platform.json"), "w") as f:
        json.dump(plat, f, indent=2)

    print("\nSummary: " + ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items())))
    print("Reports and exceptions.tsv: %s" % os.path.relpath(outdir, HERE))
    if failed:
        print("Not passing: " + " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
