"""
Benchmark: compare adaptive (RectMesh3D) vs uniform (UniformMesh3D) propagation
for a 3-port photonic lantern using lant3big.

Run with:
    python benchmarks/compare_adaptive_vs_uniform.py

Results are printed to stdout and optionally written to
benchmarks/results/benchmark_results.txt.
"""
import time
import os
import numpy as np
import lightbeam.optics as optics
from lightbeam.mesh import RectMesh3D, UniformMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import overlap_nonu

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

WL = 1.0          # wavelength [µm]
NCLAD = 1.4504
NCORE = NCLAD + 0.0088
NJACK = NCLAD - 5.5e-3

RCORE  = 2.0      # core radius [µm]
RCLAD  = 25.0     # cladding radius [µm]
OFFSET = 8.0      # core centre offset [µm]

XW = 60.0         # transverse window half-width [µm]
YW = 60.0
ZW = 100.0        # propagation length [µm]
DS = 0.5          # transverse sampling [µm]
DZ = 1.0          # axial step [µm]
PML = 4           # PML cell count


def make_lantern():
    return optics.lant3big(
        RCORE, RCLAD, NCORE, NCLAD, NJACK,
        OFFSET, ZW, final_scale=0.1
    )


def launch_field(xy):
    """Gaussian beam centred at the top core position."""
    x0, y0 = 0.0, OFFSET
    sigma = RCORE
    u = np.exp(-((xy.xg - x0)**2 + (xy.yg - y0)**2) / (2 * sigma**2))
    return u.astype(complex)


def run_adaptive():
    mesh = RectMesh3D(XW, YW, ZW, DS, DZ, PML)
    optic = make_lantern()
    prop = Prop3D(WL, mesh, optic, NCLAD)

    t0 = time.perf_counter()
    u, u0 = prop.prop2end(
        lambda xg, yg: np.exp(-(xg**2 + yg**2) / (2 * RCORE**2)).astype(complex)
    )
    elapsed = time.perf_counter() - t0
    return elapsed, mesh, u, u0


def run_uniform():
    mesh = UniformMesh3D(XW, YW, ZW, DS, DZ, PML)
    optic = make_lantern()
    prop = Prop3D(WL, mesh, optic, NCLAD)

    xy = mesh.xy
    u_in = launch_field(xy)

    t0 = time.perf_counter()
    u0 = prop.prop2end_uniform(u_in)
    elapsed = time.perf_counter() - t0
    return elapsed, mesh, u_in, u0


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Lightbeam BPM benchmark: adaptive vs uniform propagation")
    print("=" * 60)
    print(f"  Lantern : lant3big  rcore={RCORE}µm  rclad={RCLAD}µm")
    print(f"  Window  : {XW}×{YW}µm   Z={ZW}µm")
    print(f"  DS={DS}µm  DZ={DZ}µm  PML={PML}")
    print()

    print("Running adaptive mesh propagation …")
    t_adapt, mesh_a, u_adapt, u0_adapt = run_adaptive()
    w_a = mesh_a.xy.get_weights()
    p_adapt = float(overlap_nonu(u_adapt, u_adapt, w_a))
    print(f"  Elapsed : {t_adapt:.2f} s")
    print(f"  Output power : {p_adapt:.4f}")
    print()

    print("Running uniform mesh propagation …")
    t_uni, mesh_u, u_in_uni, u0_uni = run_uniform()
    w_u = mesh_u.xy.get_weights()
    p_uni = float(overlap_nonu(u0_uni, u0_uni, w_u))
    print(f"  Elapsed : {t_uni:.2f} s")
    print(f"  Output power : {p_uni:.4f}")
    print()

    speedup = t_adapt / t_uni if t_uni > 1e-9 else float("inf")
    print(f"Speed-up (adaptive / uniform) : {speedup:.2f}×")
    print()

    # Write summary
    summary_path = os.path.join(results_dir, "benchmark_results.txt")
    with open(summary_path, "w") as f:
        f.write("adaptive_time_s,uniform_time_s,speedup,adaptive_power,uniform_power\n")
        f.write(f"{t_adapt:.4f},{t_uni:.4f},{speedup:.4f},{p_adapt:.6f},{p_uni:.6f}\n")
    print(f"Results written to {summary_path}")


if __name__ == "__main__":
    main()
