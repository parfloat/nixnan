"""cuTile-style extraction of the simple HPL-AI mixed-precision solve path.

This ports the compact reference path from the ICL HPL-AI implementation:

1. generate a matrix and RHS,
2. convert inputs to the requested factor/solve precision,
3. factor A with no-pivot LU,
4. solve L*U*x=b,
5. convert x to the requested refinement precision,
6. optionally run GMRES refinement,
7. compute the scaled residual.

The cuTile backend maps conversion, no-pivot LU, and triangular solve to simple
tile kernels.  GMRES refinement stays on the host with NumPy so the cuTile
surface remains a compact NixNan/cuTile workload seed.
"""

# Derived from the ICL HPL-AI reference implementation. See HPL_AI_LICENSE.

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import time

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional GPU runtime
    cp = None

try:
    import cuda.tile as ct
except Exception:  # pragma: no cover - optional cuTile runtime
    ct = None


W = 128
LCG_A = 6364136223846793005
LCG_C = 1
UINT64_MASK = (1 << 64) - 1
LCG_MUL = float(np.float32(5.4210108624275222e-20))
RESIDUAL_THRESHOLD = 16.0
SUPPORTED_PRECISIONS = ("float16", "float32", "float64")
SUPPORTED_MATRIX_KINDS = ("hpl-ai", "conditioned")


@dataclass(frozen=True)
class PrecisionConfig:
    input_precision: str = "float64"
    factor_precision: str = "float32"
    solve_precision: str = "float32"
    refinement_precision: str = "float64"
    residual_precision: str = "float64"


@dataclass(frozen=True)
class MatrixConfig:
    kind: str = "hpl-ai"
    condition_number: float = 1.0e4
    seed: int = 1


@dataclass
class SolveResult:
    backend: str
    n: int
    matrix_kind: str
    condition_number: float | None
    precisions: PrecisionConfig
    elapsed_s: float
    x: np.ndarray
    scaled_residual: float
    passed: bool
    gmres_iterations: int = 0


def _validate_precision(name: str) -> str:
    if name not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision {name!r}; choose one of {SUPPORTED_PRECISIONS}")
    return name


def _np_dtype(precision: str) -> np.dtype:
    _validate_precision(precision)
    return np.dtype(precision)


def _np_linalg_dtype(precision: str) -> np.dtype:
    _validate_precision(precision)
    if precision == "float16":
        return np.dtype("float32")
    return np.dtype(precision)


def _cp_dtype(precision: str):
    if cp is None:
        raise RuntimeError("cupy is required for cuTile dtype selection")
    _validate_precision(precision)
    return {
        "float16": cp.float16,
        "float32": cp.float32,
        "float64": cp.float64,
    }[precision]


def _flat_index(row: int, col: int, n: int) -> int:
    return row + col * n


def _as_column_major_matrix(flat: np.ndarray, n: int) -> np.ndarray:
    return flat.reshape((n, n), order="F")


def _lcg_values(count: int, seed: int, input_precision: str) -> np.ndarray:
    dtype = _np_dtype(input_precision)
    state = seed & UINT64_MASK
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        state = (state * LCG_A + LCG_C) & UINT64_MASK
        out[i] = float(state) * LCG_MUL - 0.5
    return out.astype(dtype)


def hpl_ai_matrix(n: int, seed: int = 1, input_precision: str = "float64") -> np.ndarray:
    """Return the HPL-AI reference matrix in 1-D column-major layout."""
    flat = _lcg_values(n * n, seed, input_precision)
    work = flat.astype(np.float64, copy=True)
    mat = _as_column_major_matrix(work, n)
    row_abs_sums = np.sum(np.abs(mat), axis=1)
    diag = np.diag_indices(n)
    mat[diag] = row_abs_sums - np.abs(mat[diag])
    return work.astype(_np_dtype(input_precision))


def conditioned_matrix(
    n: int,
    seed: int = 1,
    condition_number: float = 1.0e4,
    input_precision: str = "float64",
) -> np.ndarray:
    """Return a symmetric positive definite matrix with approximate 2-norm condition."""
    if condition_number < 1.0:
        raise ValueError("condition_number must be >= 1")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(raw)
    spectrum = np.geomspace(condition_number, 1.0, n)
    mat = (q * spectrum) @ q.T
    mat = 0.5 * (mat + mat.T)
    return np.asarray(mat, dtype=_np_dtype(input_precision)).reshape(n * n, order="F")


def make_matrix(n: int, matrix: MatrixConfig, input_precision: str) -> np.ndarray:
    if matrix.kind == "hpl-ai":
        return hpl_ai_matrix(n, matrix.seed, input_precision)
    if matrix.kind == "conditioned":
        return conditioned_matrix(n, matrix.seed, matrix.condition_number, input_precision)
    raise ValueError(f"unsupported matrix kind {matrix.kind!r}; choose one of {SUPPORTED_MATRIX_KINDS}")


def make_rhs(n: int, seed: int = 2, input_precision: str = "float64") -> np.ndarray:
    return _lcg_values(n, seed, input_precision)


def lu_nopiv_cpu(a_input: np.ndarray, n: int, factor_precision: str = "float32") -> np.ndarray:
    lu = a_input.astype(_np_dtype(factor_precision), copy=True)
    lu_mat = _as_column_major_matrix(lu, n)
    for k in range(n - 1):
        pivot = lu_mat[k, k]
        lu_mat[k + 1 :, k] /= pivot
        lu_mat[k + 1 :, k + 1 :] -= np.outer(lu_mat[k + 1 :, k], lu_mat[k, k + 1 :])
    return lu


def lu_solve_cpu(
    lu_factor: np.ndarray,
    b_input: np.ndarray,
    n: int,
    solve_precision: str = "float32",
) -> np.ndarray:
    lu = _as_column_major_matrix(lu_factor, n)
    x = b_input.astype(_np_dtype(solve_precision), copy=True)

    for k in range(n):
        x[k + 1 :] -= x[k] * lu[k + 1 :, k]

    for k in range(n - 1, -1, -1):
        x[k] /= lu[k, k]
        x[:k] -= x[k] * lu[:k, k]

    return x


def lu_solve_preconditioner(
    lu_factor: np.ndarray,
    rhs: np.ndarray,
    n: int,
    refinement_precision: str = "float64",
) -> np.ndarray:
    dtype = _np_dtype(refinement_precision)
    lu = _as_column_major_matrix(lu_factor.astype(dtype, copy=False), n)
    x = rhs.astype(dtype, copy=True)

    for k in range(n):
        x[k + 1 :] -= x[k] * lu[k + 1 :, k]

    for k in range(n - 1, -1, -1):
        x[k] /= lu[k, k]
        x[:k] -= x[k] * lu[:k, k]

    return x


def scaled_residual(
    a_input: np.ndarray,
    x_input: np.ndarray,
    b_input: np.ndarray,
    n: int,
    residual_precision: str = "float64",
) -> float:
    dtype = _np_dtype(residual_precision)
    a = _as_column_major_matrix(a_input.astype(dtype, copy=False), n)
    x = x_input.astype(dtype, copy=False)
    b = b_input.astype(dtype, copy=False)
    residual = a @ x - b
    norm_a = np.linalg.norm(a, ord=np.inf)
    norm_x = np.linalg.norm(x, ord=np.inf)
    norm_b = np.linalg.norm(b, ord=np.inf)
    eps = np.finfo(dtype).eps / dtype.type(2.0)
    denom = eps * (norm_a * norm_x + norm_b) * dtype.type(n)
    return float(np.linalg.norm(residual, ord=np.inf) / denom)


def _rotmat(a: float, b: float) -> tuple[float, float]:
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        temp = a / b
        sn = 1.0 / math.sqrt(1.0 + temp * temp)
        cs = temp * sn
        return cs, sn
    temp = b / a
    cs = 1.0 / math.sqrt(1.0 + temp * temp)
    sn = temp * cs
    return cs, sn


def _solve_small_upper(h: np.ndarray, s: np.ndarray, refinement_precision: str) -> np.ndarray:
    dtype = _np_dtype(refinement_precision)
    linalg_dtype = _np_linalg_dtype(refinement_precision)
    h_work = h.astype(linalg_dtype, copy=False)
    s_work = s.astype(linalg_dtype, copy=False)
    try:
        return np.linalg.solve(h_work, s_work).astype(dtype, copy=False)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(h_work, s_work, rcond=None)[0].astype(dtype, copy=False)


def gmres_refine(
    a_input: np.ndarray,
    x_initial: np.ndarray,
    b_input: np.ndarray,
    lu_factor: np.ndarray,
    n: int,
    max_iter: int,
    refinement_precision: str = "float64",
    residual_precision: str = "float64",
) -> tuple[np.ndarray, int]:
    """Reference-style left-preconditioned GMRES refinement."""
    dtype = _np_dtype(refinement_precision)
    a = _as_column_major_matrix(a_input.astype(dtype, copy=False), n)
    b = b_input.astype(dtype, copy=False)
    x = x_initial.astype(dtype, copy=True)
    old_x = x.copy()
    m = min(max_iter, n)
    tol = np.finfo(dtype).eps / dtype.type(2.0) / (dtype.type(n) / dtype.type(4.0))

    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        norm_b = dtype.type(1.0)

    r = lu_solve_preconditioner(lu_factor, b - a @ x, n, refinement_precision)
    if np.linalg.norm(r) / norm_b < tol:
        return x, 0

    cs = np.zeros(m, dtype=dtype)
    sn = np.zeros(m, dtype=dtype)
    s = np.zeros(m + 1, dtype=dtype)
    h = np.zeros((m + 1, m), dtype=dtype)
    v = np.zeros((n, m + 1), dtype=dtype)
    iterations = 0

    norm_r = np.linalg.norm(r)
    if norm_r == 0.0:
        return x, 0
    v[:, 0] = r / norm_r
    s[0] = norm_r

    for i in range(m):
        iterations = i + 1
        w = lu_solve_preconditioner(lu_factor, a @ v[:, i], n, refinement_precision)

        for k in range(i + 1):
            h[k, i] = np.dot(w, v[:, k])
            w -= h[k, i] * v[:, k]

        h[i + 1, i] = np.linalg.norm(w)
        if h[i + 1, i] != 0.0 and i + 1 < m + 1:
            v[:, i + 1] = w / h[i + 1, i]

        for k in range(i):
            temp = cs[k] * h[k, i] + sn[k] * h[k + 1, i]
            h[k + 1, i] = -sn[k] * h[k, i] + cs[k] * h[k + 1, i]
            h[k, i] = temp

        cs_i, sn_i = _rotmat(float(h[i, i]), float(h[i + 1, i]))
        cs[i] = cs_i
        sn[i] = sn_i
        temp = cs[i] * s[i]
        s[i + 1] = -sn[i] * s[i]
        s[i] = temp
        h[i, i] = cs[i] * h[i, i] + sn[i] * h[i + 1, i]
        h[i + 1, i] = dtype.type(0.0)

        error = abs(float(s[i + 1] / norm_b))
        if error <= float(tol):
            y = _solve_small_upper(h[: i + 1, : i + 1], s[: i + 1], refinement_precision)
            candidate = x + v[:, : i + 1] @ y
            if scaled_residual(a_input, candidate, b_input, n, residual_precision) <= RESIDUAL_THRESHOLD:
                return candidate, iterations
            x = old_x.copy()

    y = _solve_small_upper(h[:m, :m], s[:m], refinement_precision)
    x = x + v[:, :m] @ y
    return x, iterations


def run_cpu(
    n: int,
    matrix: MatrixConfig,
    precisions: PrecisionConfig,
    max_iter: int = 50,
    refine: bool = True,
) -> SolveResult:
    a = make_matrix(n, matrix, precisions.input_precision)
    b = make_rhs(n, matrix.seed + 1, precisions.input_precision)

    start = time.perf_counter()
    lu = lu_nopiv_cpu(a, n, precisions.factor_precision)
    x = lu_solve_cpu(lu, b, n, precisions.solve_precision).astype(
        _np_dtype(precisions.refinement_precision),
        copy=False,
    )
    iterations = 0
    if refine:
        x, iterations = gmres_refine(
            a,
            x,
            b,
            lu,
            n,
            max_iter,
            precisions.refinement_precision,
            precisions.residual_precision,
        )
    elapsed = time.perf_counter() - start

    error = scaled_residual(a, x, b, n, precisions.residual_precision)
    backend = "cpu-reference+gmres" if refine else "cpu-reference"
    condition = matrix.condition_number if matrix.kind == "conditioned" else None
    return SolveResult(backend, n, matrix.kind, condition, precisions, elapsed, x, error, error < RESIDUAL_THRESHOLD, iterations)


if ct is not None:

    @ct.kernel  # type: ignore[misc]
    def convert_to_f16_kernel(src, dst, total: ct.Constant[int]):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int32)
        gid = pid * W + lane
        tile = ct.load(src, (pid,), shape=(W,), padding_mode=ct.PaddingMode.ZERO)
        out = ct.astype(tile, ct.float16)
        out = ct.where(gid < total, out, 0.0)
        ct.store(dst, (pid,), out)


    @ct.kernel  # type: ignore[misc]
    def convert_to_f32_kernel(src, dst, total: ct.Constant[int]):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int32)
        gid = pid * W + lane
        tile = ct.load(src, (pid,), shape=(W,), padding_mode=ct.PaddingMode.ZERO)
        out = ct.astype(tile, ct.float32)
        out = ct.where(gid < total, out, 0.0)
        ct.store(dst, (pid,), out)


    @ct.kernel  # type: ignore[misc]
    def convert_to_f64_kernel(src, dst, total: ct.Constant[int]):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int32)
        gid = pid * W + lane
        tile = ct.load(src, (pid,), shape=(W,), padding_mode=ct.PaddingMode.ZERO)
        out = ct.astype(tile, ct.float64)
        out = ct.where(gid < total, out, 0.0)
        ct.store(dst, (pid,), out)


    @ct.kernel  # type: ignore[misc]
    def scale_column_kernel(lu, k: ct.Constant[int], n: ct.Constant[int], total: ct.Constant[int]):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int64)
        offset = pid * W + lane
        active = offset < total
        row = k + 1 + offset

        pivot = ct.load(lu, (k + k * n,), shape=())
        index = row + k * n
        value = ct.gather(lu, index, mask=active, padding_value=0.0)
        ct.scatter(lu, index, value / pivot, mask=active)


    @ct.kernel  # type: ignore[misc]
    def rank1_update_kernel(
        lu,
        k: ct.Constant[int],
        n: ct.Constant[int],
        trailing: ct.Constant[int],
        total: ct.Constant[int],
    ):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int64)
        flat = pid * W + lane
        active = flat < total

        row = k + 1 + (flat % trailing)
        col = k + 1 + (flat // trailing)
        lhs = ct.gather(lu, row + k * n, mask=active, padding_value=0.0)
        rhs = ct.gather(lu, k + col * n, mask=active, padding_value=0.0)
        index = row + col * n
        cur = ct.gather(lu, index, mask=active, padding_value=0.0)
        ct.scatter(lu, index, cur - lhs * rhs, mask=active)


    @ct.kernel  # type: ignore[misc]
    def forward_subtract_kernel(
        lu,
        rhs,
        k: ct.Constant[int],
        n: ct.Constant[int],
        total: ct.Constant[int],
    ):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int64)
        offset = pid * W + lane
        active = offset < total
        row = k + 1 + offset

        xk = ct.load(rhs, (k,), shape=())
        coeff = ct.gather(lu, row + k * n, mask=active, padding_value=0.0)
        value = ct.gather(rhs, row, mask=active, padding_value=0.0)
        ct.scatter(rhs, row, value - xk * coeff, mask=active)


    @ct.kernel  # type: ignore[misc]
    def scale_rhs_kernel(lu, rhs, k: ct.Constant[int], n: ct.Constant[int]):
        pivot = ct.load(lu, (k + k * n,), shape=())
        xk = ct.load(rhs, (k,), shape=())
        ct.store(rhs, (k,), xk / pivot)


    @ct.kernel  # type: ignore[misc]
    def backward_subtract_kernel(
        lu,
        rhs,
        k: ct.Constant[int],
        n: ct.Constant[int],
        total: ct.Constant[int],
    ):
        pid = ct.bid(0)
        lane = ct.arange(W, dtype=ct.int64)
        row = pid * W + lane
        active = row < total

        xk = ct.load(rhs, (k,), shape=())
        coeff = ct.gather(lu, row + k * n, mask=active, padding_value=0.0)
        value = ct.gather(rhs, row, mask=active, padding_value=0.0)
        ct.scatter(rhs, row, value - xk * coeff, mask=active)


def _conversion_kernel(precision: str):
    _validate_precision(precision)
    return {
        "float16": convert_to_f16_kernel,
        "float32": convert_to_f32_kernel,
        "float64": convert_to_f64_kernel,
    }[precision]


def _grid_for(total: int) -> tuple[int, int, int]:
    return ((total + W - 1) // W, 1, 1)


def run_cutile(
    n: int,
    matrix: MatrixConfig,
    precisions: PrecisionConfig,
    max_iter: int = 50,
    refine: bool = True,
) -> SolveResult:
    if cp is None:
        raise RuntimeError("cupy is required for --backend cutile")
    if ct is None:
        raise RuntimeError("cuda.tile is required for --backend cutile")

    a = make_matrix(n, matrix, precisions.input_precision)
    b = make_rhs(n, matrix.seed + 1, precisions.input_precision)
    total_matrix = n * n
    stream = cp.cuda.get_current_stream()

    a_gpu = cp.asarray(a, dtype=_cp_dtype(precisions.input_precision))
    b_gpu = cp.asarray(b, dtype=_cp_dtype(precisions.input_precision))
    lu_gpu = cp.empty(total_matrix, dtype=_cp_dtype(precisions.factor_precision))
    rhs_gpu = cp.empty(n, dtype=_cp_dtype(precisions.solve_precision))
    x_gpu = cp.empty(n, dtype=_cp_dtype(precisions.refinement_precision))

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    ct.launch(stream, _grid_for(total_matrix), _conversion_kernel(precisions.factor_precision), (a_gpu, lu_gpu, total_matrix))
    ct.launch(stream, _grid_for(n), _conversion_kernel(precisions.solve_precision), (b_gpu, rhs_gpu, n))

    for k in range(n - 1):
        below = n - k - 1
        ct.launch(stream, _grid_for(below), scale_column_kernel, (lu_gpu, k, n, below))
        ct.launch(
            stream,
            _grid_for(below * below),
            rank1_update_kernel,
            (lu_gpu, k, n, below, below * below),
        )

    for k in range(n):
        below = n - k - 1
        if below:
            ct.launch(stream, _grid_for(below), forward_subtract_kernel, (lu_gpu, rhs_gpu, k, n, below))

    for k in range(n - 1, -1, -1):
        ct.launch(stream, (1, 1, 1), scale_rhs_kernel, (lu_gpu, rhs_gpu, k, n))
        if k:
            ct.launch(stream, _grid_for(k), backward_subtract_kernel, (lu_gpu, rhs_gpu, k, n, k))

    ct.launch(stream, _grid_for(n), _conversion_kernel(precisions.refinement_precision), (rhs_gpu, x_gpu, n))
    end.record()
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0

    x = cp.asnumpy(x_gpu)
    iterations = 0
    if refine:
        lu = cp.asnumpy(lu_gpu)
        x, iterations = gmres_refine(
            a,
            x,
            b,
            lu,
            n,
            max_iter,
            precisions.refinement_precision,
            precisions.residual_precision,
        )
    error = scaled_residual(a, x, b, n, precisions.residual_precision)
    backend = "cutile+host-gmres" if refine else "cutile"
    condition = matrix.condition_number if matrix.kind == "conditioned" else None
    return SolveResult(backend, n, matrix.kind, condition, precisions, elapsed, x, error, error < RESIDUAL_THRESHOLD, iterations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPL-AI no-pivot LU extraction in cuTile notation")
    parser.add_argument("--backend", choices=("cpu", "cutile"), default="cpu")
    parser.add_argument("--n", type=int, default=32, help="linear system size")
    parser.add_argument("--matrix-kind", choices=SUPPORTED_MATRIX_KINDS, default="hpl-ai")
    parser.add_argument("--condition-number", type=float, default=1.0e4)
    parser.add_argument("--max-iter", type=int, default=50, help="maximum GMRES iterations")
    parser.add_argument("--seed", type=int, default=1, help="matrix generator seed")
    parser.add_argument("--input-precision", choices=SUPPORTED_PRECISIONS, default="float64")
    parser.add_argument("--factor-precision", choices=SUPPORTED_PRECISIONS, default="float32")
    parser.add_argument("--solve-precision", choices=SUPPORTED_PRECISIONS, default="float32")
    parser.add_argument("--refinement-precision", choices=SUPPORTED_PRECISIONS, default="float64")
    parser.add_argument("--residual-precision", choices=SUPPORTED_PRECISIONS, default="float64")
    parser.add_argument("--no-gmres", action="store_true", help="stop after the mixed-precision LU solve")
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="also run the CPU extraction and report max |x_backend-x_cpu|",
    )
    return parser.parse_args()


def _print_result(result: SolveResult) -> None:
    print(f"backend={result.backend}")
    print(f"n={result.n}")
    print(f"matrix_kind={result.matrix_kind}")
    print(f"condition_number={result.condition_number if result.condition_number is not None else 'n/a'}")
    print(f"input_precision={result.precisions.input_precision}")
    print(f"factor_precision={result.precisions.factor_precision}")
    print(f"solve_precision={result.precisions.solve_precision}")
    print(f"refinement_precision={result.precisions.refinement_precision}")
    print(f"residual_precision={result.precisions.residual_precision}")
    print(f"elapsed_s={result.elapsed_s:.6f}")
    print(f"gmres_iterations={result.gmres_iterations}")
    print(f"scaled_residual={result.scaled_residual:.6e}")
    print(f"passes_hpl_ai_threshold={str(result.passed).lower()}")


def main() -> None:
    args = parse_args()
    if args.n < 2:
        raise ValueError("--n must be at least 2")

    precisions = PrecisionConfig(
        input_precision=args.input_precision,
        factor_precision=args.factor_precision,
        solve_precision=args.solve_precision,
        refinement_precision=args.refinement_precision,
        residual_precision=args.residual_precision,
    )
    matrix = MatrixConfig(args.matrix_kind, args.condition_number, args.seed)
    max_iter = min(args.max_iter, args.n - 1)
    refine = not args.no_gmres
    result = (
        run_cpu(args.n, matrix, precisions, max_iter, refine)
        if args.backend == "cpu"
        else run_cutile(args.n, matrix, precisions, max_iter, refine)
    )
    _print_result(result)

    if args.compare_cpu and args.backend != "cpu":
        cpu_result = run_cpu(args.n, matrix, precisions, max_iter, refine)
        max_abs = float(np.max(np.abs(result.x.astype(np.float64) - cpu_result.x.astype(np.float64))))
        print(f"max_abs_diff_vs_cpu={max_abs:.6e}")


if __name__ == "__main__":
    main()
