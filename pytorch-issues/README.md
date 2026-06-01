# pytorch-issues — NixNan diagnostics for PyTorch FP-exception bugs

This branch collects PyTorch bug reports for which we ran a minimal
reproducer under [NixNan](https://github.com/parfloat/nixnan/) and
packaged the trace, our analysis, and a one-script reproduction
recipe — one folder per issue.

## Layout

```
pytorch-issues/
  issue<NNN>/
    README.md           gist of the bug + how to reproduce
    data/
      repro.py          minimal Python reproducer
      _diag.py          shared diagnostic helper
      bin_spec.json     per-binade histogram spec
      nixnan.nnlog      our captured trace
      stdout.nnlog      our captured Python output
      issueFeedback.md  our reading of the trace + tentative recommendation
    usingNixNan/
      reproduce.sh      build NixNan + run the repro on your machine
```

## How to use

Pick the issue you care about:

```bash
git checkout pytorch-issues
cd pytorch-issues/issue<NNN>
cat README.md
./usingNixNan/reproduce.sh
```

The script builds NixNan from this repo (one-time) if it isn't already
built, then runs the reproducer under the canonical sweep profile
(`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`).

## Broader context

This is a curated subset of a larger 50-repro corpus. The full corpus
— including sweep history (peek1.md through peek8.md), plan documents
(Plan2.md through Plan5_next10.md), and a 50-repro ranking with a
HARMFUL / HARMLESS verdict column (`kernel_summary.md`) — lives in
[parfloat/parfloat-class](https://github.com/parfloat/parfloat-class)
under `pytorch-nixnan/`.

## What's in each issueFeedback.md

For each issue, we wrote a short markdown with:

- A factual summary of what NixNan observed (verdict, per-precision
  exception counts, top kernels emitting events, output state).
- A deliberately tentative "Recommendation" paragraph — framed as
  "this is what the trace suggests; here's what to verify."
- Three modest questions back to the original issue poster:
  1. Does this trace plus our reading help narrow the root cause for
     you now?
  2. Would such a trace have helped at the time you filed the issue?
  3. Was this issue a show-stopper for your work, or filed primarily
     as a best-practice / correctness flag?
