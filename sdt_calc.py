"""
sdt_calc.py  —  Spin Diffusion Tensor Calculator

New in this version:
  - ShiftML3 integration: predicts per-site isotropic chemical shielding
    and the full 3×3 CSA tensor directly from the crystal structure.
  - cs_source = "shiftml" : use ML-predicted shifts + CSA in the SDT.
  - cs_source = "manual"  : use user-supplied ics / delta / eta scalars
    (same as the original MATLAB code).
  - cs_source = "none"    : all omega_cs = 0 (original web app default).

Physics of CSA → omega_cs  (matches SDT_Calc.m exactly)
  Crystalline model  : omega_cs[i] = ics[i] * nu0 * 1e-6   (Hz)
  Amorphous model    : random PAS orientation drawn per spin,
      ppm_cs[i] = ics[i] + (delta[i]/2) * (3cos²θ−1 + η*sin²θ*cos2φ)
      omega_cs[i] = ppm_cs[i] * nu0 * 1e-6

ShiftML note
  ShiftML predicts absolute shielding σ (ppm).  Only *differences* between
  sites enter the ZQ spectral density  δν_ij = ½(ω_i − ω_j), so the unknown
  reference offset cancels exactly.  The Haeberlen CSA parameters are
  extracted from the diagonalised shielding tensor.
"""

import re
import warnings
import numpy as np
from itertools import product as iproduct

# =============================================================================
#  NUCLEUS PARAMETERS
# =============================================================================
NUCLEUS_PARAMS = {
    "H": {"gyro": 267.522e6, "abund": 0.99985},
    "F": {"gyro": 251.815e6, "abund": 1.0},
    "P": {"gyro": 108.291e6, "abund": 1.0},
    "C": {"gyro":  67.283e6, "abund": 0.0107},
}

# Elements supported by ShiftML3
SHIFTML_ELEMENTS = {"H", "C", "N", "O", "S", "F", "P", "Cl", "Na", "Ca", "Mg", "K"}

# =============================================================================
#  SHIFTML INTEGRATION
# =============================================================================

def run_shiftml(cif_text: str, nucleus: str):
    """
    Run ShiftML3 in an isolated subprocess with gevent-friendly polling.

    subprocess.run() blocks the entire gevent hub (it calls select() internally),
    which prevents Gunicorn's heartbeat and triggers WORKER TIMEOUT even under
    gevent monkey-patching.  The fix: use Popen and poll() in a loop with
    gevent.sleep() so the hub can service other greenlets while we wait.

    Returns (cs_iso, cs_delta, cs_eta, atom_idx) on success.
    Raises ValueError with a clear message on failure.
    """
    import subprocess, sys, json as _json, tempfile, os, time

    # Try to import gevent.sleep for cooperative yielding; fall back to time.sleep
    try:
        from gevent import sleep as gsleep
    except ImportError:
        from time import sleep as gsleep

    # Write CIF to a temp file for the subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif",
                                     delete=False, encoding="utf-8") as tf:
        tf.write(cif_text)
        cif_path = tf.name

    worker_script = f"""
import sys, json, subprocess, numpy as np

# Install dependencies at runtime if not present (keeps build lean)
def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

ensure('ase')
ensure('shiftml')

cif_path = {repr(cif_path)}
nucleus  = {repr(nucleus)}
try:
    from ase.io import read as ase_read
    from shiftml.ase import ShiftML

    frame = ase_read(cif_path, format='cif')
    symbols = np.array(frame.get_chemical_symbols())
    atom_idx = np.where(symbols == nucleus)[0]
    if len(atom_idx) == 0:
        print(json.dumps({{'error': f"No {{nucleus}} atoms found in CIF by ASE."}}))
        sys.exit(0)

    calculator = ShiftML("ShiftML3", device="cpu")
    cs_tensor_all = calculator.get_cs_tensor(frame)
    tensors = cs_tensor_all[atom_idx]

    cs_iso_l, cs_delta_l, cs_eta_l = [], [], []
    for T in tensors:
        eigs = np.linalg.eigvalsh(T)
        s11, s22, s33 = eigs[0], eigs[1], eigs[2]
        s_iso = (s11 + s22 + s33) / 3.0
        devs = sorted([(abs(v - s_iso), v) for v in [s11, s22, s33]], reverse=True)
        s_ZZ = devs[0][1]; s_YY = devs[2][1]; s_XX = devs[1][1]
        delta_i = s_ZZ - s_iso
        eta_i = abs((s_YY - s_XX) / delta_i) if abs(delta_i) > 1e-6 else 0.0
        eta_i = max(0.0, min(1.0, eta_i))
        cs_iso_l.append(float(s_iso))
        cs_delta_l.append(float(delta_i))
        cs_eta_l.append(float(eta_i))

    print(json.dumps({{
        'iso':   cs_iso_l,
        'delta': cs_delta_l,
        'eta':   cs_eta_l,
        'idx':   atom_idx.tolist(),
    }}))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""

    TIMEOUT = 150   # seconds — generous for model download + inference
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.time() + TIMEOUT
        while proc.poll() is None:
            if time.time() > deadline:
                proc.kill()
                raise ValueError(
                    f"ShiftML3 timed out after {TIMEOUT}s. "
                    "The free-tier server is too memory-constrained for this model. "
                    "Try a different nucleus or use Manual / None CS mode."
                )
            gsleep(0.5)   # yield to gevent hub — keeps SSE alive and heartbeat running

        stdout, stderr = proc.communicate()

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"ShiftML3 subprocess error: {e}")
    finally:
        try:
            os.unlink(cif_path)
        except OSError:
            pass
        if proc and proc.poll() is None:
            proc.kill()

    # Parse the last non-empty line of stdout as JSON
    stdout_lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    if not stdout_lines:
        snippet = stderr[-400:] if stderr else "(no stderr)"
        raise ValueError(
            "ShiftML3 produced no output — likely killed by OOM on the free tier. "
            f"stderr: {snippet}"
        )

    try:
        result = _json.loads(stdout_lines[-1])
    except _json.JSONDecodeError:
        raise ValueError(f"ShiftML3 output unparseable: {stdout_lines[-1][:200]}")

    if "error" in result:
        raise ValueError(f"ShiftML3: {result['error']}")

    cs_iso   = np.array(result["iso"],   dtype=float)
    cs_delta = np.array(result["delta"], dtype=float)
    cs_eta   = np.array(result["eta"],   dtype=float)
    atom_idx = np.array(result["idx"],   dtype=int)
    return cs_iso, cs_delta, cs_eta, atom_idx
    cs_eta   = np.array(result["eta"],   dtype=float)
    atom_idx = np.array(result["idx"],   dtype=int)

    return cs_iso, cs_delta, cs_eta, atom_idx


# =============================================================================
#  OMEGA_CS CONSTRUCTION   (matches SDT_Calc.m exactly)
# =============================================================================

def build_omega_cs(N, nu0, disorder,
                   ics_arr, delta_arr, eta_arr):
    """
    Compute the per-spin chemical-shift frequency vector omega_cs (Hz).

    Parameters
    ----------
    N         : number of spin sites
    nu0       : Larmor frequency (Hz)
    disorder  : positional disorder amplitude (0 = crystalline)
    ics_arr   : (N,) isotropic shielding/shift per site (ppm)
    delta_arr : (N,) Haeberlen anisotropy per site (ppm)
    eta_arr   : (N,) asymmetry per site [0–1]

    Returns
    -------
    omega_cs  : (N,) array in Hz
    """
    if disorder > 0:
        # Amorphous model: draw random PAS orientations (matches MATLAB exactly)
        phi   = 2 * np.pi * np.random.rand(N)
        theta = np.arccos(2 * np.random.rand(N) - 1)
        # ppm_cs[i] = ics[i] + (delta[i]/2)*(3cos²θ-1 + η*sin²θ*cos2φ)
        ppm_cs = (ics_arr
                  + (delta_arr / 2) * (3 * np.cos(theta)**2 - 1
                                       + eta_arr * np.sin(theta)**2 * np.cos(2 * phi)))
    else:
        # Crystalline model: isotropic shift only
        ppm_cs = ics_arr.copy()

    return ppm_cs * nu0 * 1e-6   # ppm → Hz


# =============================================================================
#  PUBLIC ENTRY POINT
# =============================================================================

def run_calculation(cif_text, nucleus="F", N_wanted=1000, B0_field=9.4,
                    disorder=0.0, dist_min=2e-10, num_orientations=50,
                    mas_rate_khz=0.0, abund_mode="natural", abund_pct=None,
                    cs_source="none",
                    cs_ics=0.0, cs_delta=0.0, cs_eta=0.0):
    """
    Generator. Yields SSE progress strings then the final result dict.

    Parameters
    ----------
    cs_source   : "none"     → omega_cs = 0 everywhere (fastest)
                  "manual"   → use cs_ics, cs_delta, cs_eta scalars for all sites
                  "shiftml"  → predict per-site shielding + CSA with ShiftML3
    cs_ics      : isotropic chemical shift (ppm) — used in "manual" mode
    cs_delta    : CSA half-width / Haeberlen anisotropy δ (ppm) — manual mode
    cs_eta      : CSA asymmetry η [0–1] — manual mode
    """
    if nucleus not in NUCLEUS_PARAMS:
        raise ValueError(f"Nucleus '{nucleus}' not recognised: {list(NUCLEUS_PARAMS)}")

    gyro  = NUCLEUS_PARAMS[nucleus]["gyro"]
    nu0   = gyro * B0_field / (2 * np.pi)   # Larmor frequency (Hz)

    # Isotopic abundance
    if abund_mode == "natural":
        abund = NUCLEUS_PARAMS[nucleus]["abund"]
    else:
        if abund_pct is None:
            raise ValueError("abund_pct required when abund_mode='custom'")
        abund = max(0.0, min(1.0, float(abund_pct) / 100.0))

    omega_r = mas_rate_khz * 1e3
    Na      = 6.02214076e23
    B0      = np.array([0.0, 0.0, 1.0])

    # -------------------------------------------------------------------------
    # Step 1: ShiftML prediction (before CIF replication, on unique sites)
    # -------------------------------------------------------------------------
    shiftml_result = None
    shiftml_error  = None

    if cs_source == "shiftml":
        yield "status:shiftml:Running ShiftML3 chemical shielding prediction…"
        try:
            sm_iso, sm_delta, sm_eta, sm_idx = run_shiftml(cif_text, nucleus)
            shiftml_result = {
                "iso":   sm_iso,    # (M,) ppm — unique sites
                "delta": sm_delta,  # (M,) ppm
                "eta":   sm_eta,    # (M,)
            }
            yield (f"status:shiftml:ShiftML3 done — {len(sm_iso)} unique {nucleus} sites "
                   f"predicted (δ_iso range: {sm_iso.min():.1f}–{sm_iso.max():.1f} ppm)")
        except Exception as e:
            shiftml_error = str(e)
            yield f"status:shiftml_error:{shiftml_error}"
            # Fall back to zero shifts so calculation still runs
            cs_source = "none"

    # -------------------------------------------------------------------------
    # Step 2: Parse CIF → Cartesian coordinates
    # -------------------------------------------------------------------------
    coords, cell, unique_coords = parse_cif(cif_text, nucleus, N_wanted, abund)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    N = len(x)
    if N < 2:
        raise ValueError("Fewer than 2 spin sites found. Check CIF and nucleus.")

    # -------------------------------------------------------------------------
    # Step 3: Apply disorder
    # -------------------------------------------------------------------------
    if disorder > 0:
        x, y, z = apply_disorder(x, y, z, N, disorder,
                                 cell["a"], cell["b"], cell["c"], dist_min)

    # -------------------------------------------------------------------------
    # Step 4: Build per-spin CSA arrays
    # -------------------------------------------------------------------------
    # The replicated structure has N sites; unique_coords has M sites.
    # We tile the ShiftML predictions to match the replicated N sites.

    if cs_source == "none":
        ics_arr   = np.zeros(N)
        delta_arr = np.zeros(N)
        eta_arr   = np.zeros(N)

    elif cs_source == "manual":
        ics_arr   = np.full(N, float(cs_ics))
        delta_arr = np.full(N, float(cs_delta))
        eta_arr   = np.full(N, float(cs_eta))

    elif cs_source == "shiftml" and shiftml_result is not None:
        # Tile the M unique-site predictions across N replicated sites.
        # The replication produces sites in blocks of M, so tile accordingly.
        M_unique = len(shiftml_result["iso"])
        repeats  = int(np.ceil(N / M_unique))
        ics_arr   = np.tile(shiftml_result["iso"],   repeats)[:N]
        delta_arr = np.tile(shiftml_result["delta"], repeats)[:N]
        eta_arr   = np.tile(shiftml_result["eta"],   repeats)[:N]
    else:
        ics_arr   = np.zeros(N)
        delta_arr = np.zeros(N)
        eta_arr   = np.zeros(N)

    # Build omega_cs (Hz) using the same formula as SDT_Calc.m
    omega_cs = build_omega_cs(N, nu0, disorder, ics_arr, delta_arr, eta_arr)

    # -------------------------------------------------------------------------
    # Step 5: Concentration / Wigner-Seitz radius
    # -------------------------------------------------------------------------
    a, b, c = cell["a"], cell["b"], cell["c"]
    al, be, ga = cell["alpha"], cell["beta"], cell["gamma"]
    Lx, Ly, Lz = x.max()-x.min(), y.max()-y.min(), z.max()-z.min()
    volume = ((Lx+a/2)*(Ly+b/2)*(Lz+c/2)
              * np.sqrt(1-np.cos(al)**2-np.cos(be)**2-np.cos(ga)**2
                        +2*np.cos(al)*np.cos(be)*np.cos(ga)))
    C_sim = N / (volume * 1e3 * Na)
    wsr   = 0.1 * (3/(4*np.pi*C_sim*Na))**(1/3)

    # -------------------------------------------------------------------------
    # Step 6: Diffusion tensor — orientation averaging loop
    # -------------------------------------------------------------------------
    if num_orientations <= 1:
        result = calculate_sdt(x, y, z, B0, gyro, N, omega_cs, dist_min, omega_r)
        D_final = result["D"]
        D_iso_f, D_FA_f = result["D_iso"], result["D_FA"]
        D_par_f, D_perp_f = result["D_parallel"], result["D_perp"]
        M2_f = result["M2"]
        eig_vals_f = np.array(result["eigenvalues"])
        eig_vecs_f = np.array(result["eigenvectors"])
        orient_averaged = False

    else:
        alpha_rot = 2*np.pi*np.random.rand(num_orientations)
        beta_rot  = np.arccos(2*np.random.rand(num_orientations)-1)

        D_stack = np.zeros((3, 3, num_orientations))
        D_iso_l, D_FA_l, D_par_l, D_perp_l, M2_l = [], [], [], [], []

        for o in range(num_orientations):
            # For amorphous + CSA: redraw random PAS orientations each orientation
            if disorder > 0 and cs_source != "none":
                omega_cs_o = build_omega_cs(N, nu0, disorder, ics_arr, delta_arr, eta_arr)
            else:
                omega_cs_o = omega_cs

            ca, sa = np.cos(alpha_rot[o]), np.sin(alpha_rot[o])
            cb, sb = np.cos(beta_rot[o]),  np.sin(beta_rot[o])
            Rot = np.array([[cb*ca, -sa, sb*ca],
                            [cb*sa,  ca, sb*sa],
                            [-sb,     0,    cb]])
            B0rot = Rot @ B0
            res = calculate_sdt(x, y, z, B0rot, gyro, N, omega_cs_o, dist_min, omega_r)
            D_stack[:,:,o] = res["D"]
            D_iso_l.append(res["D_iso"]); D_FA_l.append(res["D_FA"])
            D_par_l.append(res["D_parallel"]); D_perp_l.append(res["D_perp"])
            M2_l.append(res["M2"])
            yield f"progress:{o+1}:{num_orientations}"

        D_final   = D_stack.mean(axis=2)
        D_iso_f   = float(np.mean(D_iso_l))
        D_FA_f    = float(np.mean(D_FA_l))
        D_par_f   = float(np.mean(D_par_l))
        D_perp_f  = float(np.mean(D_perp_l))
        M2_f      = float(np.mean(M2_l))
        orient_averaged = True

        eig_vals_f, eig_vecs_tmp = np.linalg.eigh(D_final)
        idx = np.argsort(eig_vals_f)
        eig_vals_f = eig_vals_f[idx]
        eig_vecs_f = eig_vecs_tmp[:, idx].T

    # -------------------------------------------------------------------------
    # Step 7: Nearest neighbours + plot data
    # -------------------------------------------------------------------------
    pts   = np.stack([x, y, z], axis=1)
    diffs = pts[:,None,:]-pts[None,:,:]
    dists = np.sqrt((diffs**2).sum(axis=2))*1e10
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    plot_idx = np.random.choice(N, min(N, 2000), replace=False)

    # CSA summary for display
    csa_info = None
    if cs_source == "shiftml" and shiftml_result is not None:
        csa_info = {
            "source":    "ShiftML3",
            "n_sites":   int(len(shiftml_result["iso"])),
            "iso_mean":  float(np.mean(shiftml_result["iso"])),
            "iso_std":   float(np.std(shiftml_result["iso"])),
            "iso_min":   float(np.min(shiftml_result["iso"])),
            "iso_max":   float(np.max(shiftml_result["iso"])),
            "delta_mean":float(np.mean(shiftml_result["delta"])),
            "eta_mean":  float(np.mean(shiftml_result["eta"])),
            "iso_values":shiftml_result["iso"].tolist(),     # per unique site
            "delta_values":shiftml_result["delta"].tolist(),
            "eta_values":  shiftml_result["eta"].tolist(),
            "error":     shiftml_error,
        }
    elif cs_source == "manual":
        csa_info = {
            "source": "manual",
            "iso_mean": float(cs_ics),
            "delta_mean": float(cs_delta),
            "eta_mean": float(cs_eta),
        }

    yield {
        "D":                    D_final.tolist(),
        "D_iso":                float(D_iso_f),
        "D_FA":                 float(D_FA_f),
        "D_parallel":           float(D_par_f),
        "D_perp":               float(D_perp_f),
        "M2":                   float(M2_f),
        "N_spins":              N,
        "concentration":        float(C_sim),
        "wigner_seitz_r":       float(wsr*1e10),
        "orientation_averaged": orient_averaged,
        "num_orientations":     num_orientations if orient_averaged else 1,
        "mas_rate_khz":         float(mas_rate_khz),
        "abund_used":           float(abund),
        "cs_source":            cs_source,
        "csa_info":             csa_info,
        "shiftml_error":        shiftml_error,
        "PAS": {
            "eigenvalues":  eig_vals_f.tolist(),
            "eigenvectors": eig_vecs_f.tolist(),
        },
        "nearest_neighbors": {
            "mean": float(nn.mean()), "min": float(nn.min()), "max": float(nn.max()),
        },
        "unit_cell":        {k: float(v) for k, v in cell.items()},
        "unique_coords":    unique_coords.tolist(),
        "cartesian_sample": {
            "x": (x[plot_idx]*1e9).tolist(),
            "y": (y[plot_idx]*1e9).tolist(),
            "z": (z[plot_idx]*1e9).tolist(),
        },
    }


# =============================================================================
#  DIFFUSION TENSOR
# =============================================================================

def calculate_sdt(x, y, z, B0, gyro, N, omega_cs, dist_min, omega_r=0.0):
    """
    Compute the 3×3 spin diffusion tensor.

    Speed optimisations vs. previous version
    -----------------------------------------
    • float32 throughout  → halves memory, ~2× faster on CPU
    • Bij_squared, T_αβ, nu_diff precomputed once outside harmonic loop
    • exp() argument clipped to [-80, 0] to avoid overflow/underflow NaNs
    • No redundant intermediate arrays
    """
    hbar = np.float32(1.05457182e-34)
    mu0  = np.float32(4 * np.pi * 1e-7)

    # Cast inputs to float32
    x32, y32, z32 = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    B0_32 = B0.astype(np.float32)
    gyro32 = np.float32(gyro)

    # --- Pairwise displacements (N×N, float32) ---
    Rx = x32[:, None] - x32[None, :]
    Ry = y32[:, None] - y32[None, :]
    Rz = z32[:, None] - z32[None, :]

    # Minimum-image periodic boundary conditions
    Lx = np.float32(x32.max() - x32.min())
    Ly = np.float32(y32.max() - y32.min())
    Lz = np.float32(z32.max() - z32.min())
    Rx -= np.round(Rx / Lx) * Lx
    Ry -= np.round(Ry / Ly) * Ly
    Rz -= np.round(Rz / Lz) * Lz

    # Distances; exclude self-pairs and sites closer than dist_min
    r2 = Rx**2 + Ry**2 + Rz**2
    Distance = np.sqrt(r2)
    Distance[Distance < np.float32(dist_min)] = np.nan

    # Angle θ between inter-spin vector and B₀
    dot      = Rx * B0_32[0] + Ry * B0_32[1] + Rz * B0_32[2]
    cosTheta = dot / Distance
    Theta    = np.arccos(np.clip(cosTheta, -1.0, 1.0))

    # Dipolar coupling B_ij (Hz) and B_ij²
    bij         = (mu0 / np.float32(4 * np.pi)) * (gyro32**2 * hbar) / Distance**3
    bij         = np.nan_to_num(bij, nan=0.0, posinf=0.0, neginf=0.0)
    Bij         = bij * (1.0 - 3.0 * np.cos(Theta)**2)
    Bij         = np.nan_to_num(Bij, nan=0.0)
    Bij_sq      = Bij**2                          # precomputed once

    M2 = float(Bij_sq.sum()) / N

    # ZQ frequency difference matrix (float32)
    omega_cs32 = omega_cs.astype(np.float32)
    nu_diff = np.float32(0.5) * (omega_cs32[:, None] - omega_cs32[None, :])  # precomputed

    # Gaussian spectral density — clipped exp to avoid NaN from huge arguments
    inv_sqrt_2piM2 = np.float32(1.0 / np.sqrt(2.0 * np.pi * M2)) if M2 > 0 else np.float32(0.0)
    inv_2M2        = np.float32(1.0 / (2.0 * M2))                  if M2 > 0 else np.float32(0.0)

    def J(delta_m):
        if M2 <= 0:
            return np.zeros_like(delta_m)
        exponent = -(delta_m**2) * inv_2M2
        np.clip(exponent, -80.0, 0.0, out=exponent)   # prevent overflow / underflow
        return inv_sqrt_2piM2 * np.exp(exponent)

    # --- Rotor-phase-averaged spatial weights T_αβ ---
    # Rotor axis at the magic angle β_m = arccos(1/√3)
    beta_m  = np.float32(np.arccos(1.0 / np.sqrt(3.0)))
    ux = np.float32(np.sin(beta_m))
    uy = np.float32(0.0)
    uz = np.float32(np.cos(beta_m))

    udotr  = ux * Rx + uy * Ry + uz * Rz
    rpar2  = udotr**2
    rperp2 = r2 - rpar2

    Txx = np.float32(0.5) * rperp2 * (1.0 - ux*ux) + rpar2 * ux*ux
    Tyy = np.float32(0.5) * rperp2 * (1.0 - uy*uy) + rpar2 * uy*uy
    Tzz = np.float32(0.5) * rperp2 * (1.0 - uz*uz) + rpar2 * uz*uz
    Txy = np.float32(0.5) * rperp2 * (-ux*uy)       + rpar2 * ux*uy
    Txz = np.float32(0.5) * rperp2 * (-ux*uz)       + rpar2 * ux*uz
    Tyz = np.float32(0.5) * rperp2 * (-uy*uz)       + rpar2 * uy*uz

    # Wigner rotation factors at β_m
    f0sq  = np.float32(((3.0*np.cos(beta_m)**2 - 1.0) / 2.0)**2)
    f1sq  = np.float32((1.5 * np.sin(beta_m) * np.cos(beta_m))**2)
    f2sq  = np.float32((0.75 * np.sin(beta_m)**2)**2)
    Sbeta = float(f0sq + 2.0*f1sq + 2.0*f2sq)

    pi4   = np.float32(np.sqrt(np.pi) / 4.0)
    omega_r32 = np.float32(omega_r)

    # Accumulate W_II over 5 rotor harmonics m ∈ {-2,-1,0,1,2}
    WII = np.zeros_like(Bij_sq)
    for m_int, f2m in ((-2, f2sq), (-1, f1sq), (0, f0sq), (1, f1sq), (2, f2sq)):
        m = np.float32(m_int)
        WII += pi4 * Bij_sq * f2m * J(nu_diff - m * omega_r32) / Sbeta

    # --- Build D tensor from T_αβ weights ---
    normD  = np.float64(1e18) / (2.0 * N)   # back to float64 for final assembly
    WII64  = WII.astype(np.float64)
    T_all  = [[Txx, Txy, Txz], [Txy, Tyy, Tyz], [Txz, Tyz, Tzz]]
    D = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            D[i, j] = normD * float((WII64 * T_all[i][j].astype(np.float64)).sum())
            D[j, i] = D[i, j]
    D = np.real(D)

    # Eigendecomposition
    eig_vals, eig_vecs = np.linalg.eigh(D)
    idx      = np.argsort(eig_vals)
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    D_iso      = np.trace(D) / 3.0
    delta_D    = eig_vals - D_iso
    denom      = np.sqrt((eig_vals**2).sum())
    D_FA       = float(np.sqrt(1.5) * np.linalg.norm(delta_D) / denom) if denom > 0 else 0.0
    D_parallel = float(B0 @ D @ B0)
    D_perp     = (np.trace(D) - D_parallel) / 2.0

    return {
        "D": D, "D_iso": float(D_iso), "D_FA": D_FA,
        "D_parallel": D_parallel, "D_perp": float(D_perp), "M2": M2,
        "eigenvalues":  eig_vals.tolist(),
        "eigenvectors": eig_vecs.T.tolist(),
    }


# =============================================================================
#  CIF PARSING
# =============================================================================

def parse_cif(cif_text, nucleus, N_wanted, abund):
    lines = cif_text.splitlines()

    def extract(pattern):
        m = re.search(pattern, cif_text)
        if not m: raise ValueError(f"Pattern not found in CIF: {pattern}")
        return float(re.sub(r'\(\d+\)','',m.group(1)))

    a = 1e-10*extract(r'_cell_length_a\s+([\d.]+(?:\(\d+\))?)')
    b = 1e-10*extract(r'_cell_length_b\s+([\d.]+(?:\(\d+\))?)')
    c = 1e-10*extract(r'_cell_length_c\s+([\d.]+(?:\(\d+\))?)')
    al = np.deg2rad(extract(r'_cell_angle_alpha\s+([\d.]+)'))
    be = np.deg2rad(extract(r'_cell_angle_beta\s+([\d.]+)'))
    ga = np.deg2rad(extract(r'_cell_angle_gamma\s+([\d.]+)'))
    cell = {"a":a,"b":b,"c":c,"alpha":al,"beta":be,"gamma":ga}

    atom_start = next((i for i,l in enumerate(lines)
                       if l.strip().startswith("_atom_site_label")), None)
    if atom_start is None:
        raise ValueError("No _atom_site_label block found.")

    atom_lines = lines[atom_start:]
    col_headers = [l.strip() for l in atom_lines if l.strip().startswith("_atom_site_")]
    fract_x_idx = next((i for i,h in enumerate(col_headers)
                        if "_atom_site_fract_x" in h), None)
    if fract_x_idx is None:
        raise ValueError("_atom_site_fract_x not found.")

    header_end = next((i for i,l in enumerate(atom_lines)
                       if not l.strip().startswith("_") and l.strip()
                       and not l.strip().startswith("#")
                       and len(l.strip())>5), len(atom_lines))

    raw_coords = []
    for line in atom_lines[header_end:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("loop_") or line.startswith("_"):
            continue
        tokens = line.split()
        if len(tokens) < fract_x_idx+3: continue
        label = tokens[0]
        if not re.match(rf'^{re.escape(nucleus)}[\d_\s(]|^{re.escape(nucleus)}$',
                        label, re.IGNORECASE):
            continue
        try:
            fx = float(re.sub(r'\(.*?\)','',tokens[fract_x_idx]))
            fy = float(re.sub(r'\(.*?\)','',tokens[fract_x_idx+1]))
            fz = float(re.sub(r'\(.*?\)','',tokens[fract_x_idx+2]))
            raw_coords.append([fx,fy,fz])
        except (ValueError,IndexError):
            continue

    if not raw_coords:
        raise ValueError(f"No '{nucleus}' sites found in CIF.")

    base_coords   = np.array(raw_coords)
    unique_coords = apply_symmetry(lines, base_coords)
    extended      = replicate_cell(unique_coords, N_wanted)

    vol_cell = (a*b*c*np.sqrt(1-np.cos(al)**2-np.cos(be)**2-np.cos(ga)**2
                               +2*np.cos(al)*np.cos(be)*np.cos(ga)))
    T = np.array([
        [a, b*np.cos(ga), c*np.cos(be)],
        [0, b*np.sin(ga), c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)],
        [0, 0,            vol_cell/(a*b*np.sin(ga))],
    ])
    cartesian = (T @ extended.T).T
    keep = np.random.rand(len(cartesian)) < abund
    cartesian = cartesian[keep]
    return cartesian, cell, unique_coords


def apply_symmetry(lines, base_coords):
    sym_idx = next((i for i,l in enumerate(lines)
                    if "_symmetry_equiv_pos_as_xyz" in l
                    or "_space_group_symop_operation_xyz" in l), None)
    if sym_idx is None:
        return np.unique(base_coords%1.0, axis=0)
    sym_ops = []
    for line in lines[sym_idx+1:]:
        line = re.sub(r'^\d+\s+','',line.strip()).replace("'","")
        if not line or line.startswith("_") or line.startswith("loop_"): break
        sym_ops.append(line)
    all_coords = []
    for op in sym_ops:
        parts = [p.strip() for p in op.split(",")]
        if len(parts)!=3: continue
        for coord in base_coords:
            xv,yv,zv = coord
            new = []
            for expr in parts:
                expr = re.sub(r'(?<![a-z])x(?![a-z])',str(xv),expr,flags=re.IGNORECASE)
                expr = re.sub(r'(?<![a-z])y(?![a-z])',str(yv),expr,flags=re.IGNORECASE)
                expr = re.sub(r'(?<![a-z])z(?![a-z])',str(zv),expr,flags=re.IGNORECASE)
                try: new.append(eval(expr))
                except: new.append(0.0)
            all_coords.append(new)
    if not all_coords:
        return np.unique(base_coords%1.0, axis=0)
    all_coords = np.array(all_coords)%1.0
    tol = 1e-4
    return np.unique(np.round(all_coords/tol)*tol, axis=0)


def replicate_cell(unique_coords, N_wanted):
    n = len(unique_coords)
    Nrep = max(0, int(np.ceil((N_wanted/n)**(1/3)))-1)
    extended = []
    for coord in unique_coords:
        for da,db,dc in iproduct(range(Nrep+1),range(Nrep+1),range(Nrep+1)):
            extended.append(coord+np.array([da,db,dc]))
    return np.unique(np.array(extended), axis=0)


def apply_disorder(x, y, z, N, disorder, a, b, c, dist_min):
    pts = np.stack([x,y,z], axis=1).copy()
    for i in range(N):
        valid = False; attempts = 0
        while not valid and attempts < 1000:
            disp = disorder*np.array([a,b,c])*(np.random.rand(3)-0.5)
            proposed = pts[i]+disp
            dists = np.linalg.norm(pts-proposed, axis=1)
            dists[i] = np.inf
            valid = np.all(dists>=dist_min); attempts += 1
        if valid: pts[i] = proposed
    return pts[:,0], pts[:,1], pts[:,2]
