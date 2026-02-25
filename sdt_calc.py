"""
sdt_calc.py — Spin Diffusion Tensor Calculator (Python port of SDT_Calc.m)

This module reads a CIF crystal structure file, replicates the unit cell,
and computes the dipolar-driven spin diffusion tensor D via the secular
Redfield approximation.

Dependencies: numpy, scipy
"""

import re
import numpy as np
from scipy.linalg import eig
from itertools import product as iproduct


# =============================================================================
#  NUCLEUS PARAMETERS
# =============================================================================

NUCLEUS_PARAMS = {
    "H": {"gyro": 267.522e6, "abund": 0.99985},
    "F": {"gyro": 251.815e6, "abund": 1.0},
    "P": {"gyro": 108.291e6, "abund": 1.0},
}


# =============================================================================
#  PUBLIC ENTRY POINT
# =============================================================================

def run_calculation(cif_text: str, nucleus: str = "F", N_wanted: int = 1000,
                    B0_field: float = 9.4, disorder: float = 0.0,
                    dist_min: float = 2e-10, num_orientations: int = 50) -> dict:
    """
    Full SDT calculation pipeline with powder orientation averaging.

    Parameters
    ----------
    cif_text         : Raw text content of the CIF file.
    nucleus          : NMR nucleus symbol: 'H', 'F', or 'P'.
    N_wanted         : Target number of spin sites to simulate.
    B0_field         : Static magnetic field strength in Tesla.
    disorder         : Positional disorder amplitude [0–1]. 0 = perfect crystal.
    dist_min         : Hard-core exclusion radius in metres.
    num_orientations : Number of random crystallite orientations to average over.
                       0 = no averaging (single B0 along z only).

    Returns
    -------
    dict with keys: D, D_iso, D_FA, D_parallel, D_perp, M2,
                    nearest_neighbors, concentration, unit_cell,
                    orientation_averaged, num_orientations
    """
    if nucleus not in NUCLEUS_PARAMS:
        raise ValueError(f"Nucleus '{nucleus}' not supported. Choose from: {list(NUCLEUS_PARAMS)}")

    gyro  = NUCLEUS_PARAMS[nucleus]["gyro"]
    abund = NUCLEUS_PARAMS[nucleus]["abund"]

    # Reference B₀ direction: along z. Each orientation rotates this.
    B0 = np.array([0.0, 0.0, 1.0])

    # Physical constants
    Na = 6.02214076e23

    # -------------------------------------------------------------------------
    # 1. Parse CIF → Cartesian spin-site coordinates
    # -------------------------------------------------------------------------
    coords, cell, unique_coords = parse_cif(cif_text, nucleus, N_wanted, abund)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    N = len(x)

    if N < 2:
        raise ValueError("Fewer than 2 spin sites found. Check the CIF file and nucleus selection.")

    # -------------------------------------------------------------------------
    # 2. Apply positional disorder (amorphous model)
    # -------------------------------------------------------------------------
    if disorder > 0:
        x, y, z = apply_disorder(x, y, z, N, disorder,
                                 cell["a"], cell["b"], cell["c"], dist_min)

    # Isotropic chemical shift — zero for all spins (no CSA in this version)
    omega_cs = np.zeros(N)

    # -------------------------------------------------------------------------
    # 3. Compute concentration and Wigner–Seitz radius
    # -------------------------------------------------------------------------
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha, beta_angle, gamma = cell["alpha"], cell["beta"], cell["gamma"]

    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    Lz = z.max() - z.min()
    volume = ((Lx + a/2) * (Ly + b/2) * (Lz + c/2)
              * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta_angle)**2 - np.cos(gamma)**2
                        + 2*np.cos(alpha)*np.cos(beta_angle)*np.cos(gamma)))

    C_sim = N / (volume * 1e3 * Na)                       # mol L⁻¹
    wsr   = 0.1 * (3 / (4 * np.pi * C_sim * Na))**(1/3)  # Wigner–Seitz radius (m)

    # -------------------------------------------------------------------------
    # 4. Compute the diffusion tensor — with or without orientation averaging
    # -------------------------------------------------------------------------
    if num_orientations <= 1:
        # --- Single orientation: B₀ along z ----------------------------------
        result = calculate_sdt(x, y, z, B0, gyro, N, omega_cs, dist_min)
        D_final    = result["D"]
        D_iso_final    = result["D_iso"]
        D_FA_final     = result["D_FA"]
        D_par_final    = result["D_parallel"]
        D_perp_final   = result["D_perp"]
        M2_final       = result["M2"]
        orient_averaged = False

    else:
        # --- Powder average over num_orientations random orientations ---------
        # Draw random Euler angles for isotropic orientation distribution
        # (same method as MATLAB: random alpha and beta from spherical distribution)
        alpha_rot = 2 * np.pi * np.random.rand(num_orientations)       # Azimuthal [0, 2π]
        beta_rot  = np.arccos(2 * np.random.rand(num_orientations) - 1) # Polar     [0, π]

        D_stack    = np.zeros((3, 3, num_orientations))
        D_iso_list = []
        D_FA_list  = []
        D_par_list = []
        D_perp_list = []
        M2_list    = []

        for o in range(num_orientations):
            # Build rotation matrix Rot = Rot_beta @ Rot_alpha
            # Rot_alpha: rotation about Z by alpha
            ca, sa = np.cos(alpha_rot[o]), np.sin(alpha_rot[o])
            Rot_alpha = np.array([[ ca, -sa, 0],
                                  [ sa,  ca, 0],
                                  [  0,   0, 1]])

            # Rot_beta: rotation about Y by beta
            cb, sb = np.cos(beta_rot[o]), np.sin(beta_rot[o])
            Rot_beta = np.array([[ cb, 0, sb],
                                 [  0, 1,  0],
                                 [-sb, 0, cb]])

            Rot   = Rot_beta @ Rot_alpha

            # Rotate B₀ into the crystal frame for this orientation
            B0rot = Rot @ B0

            # Calculate SDT for this orientation
            res = calculate_sdt(x, y, z, B0rot, gyro, N, omega_cs, dist_min)

            D_stack[:, :, o] = res["D"]
            D_iso_list.append(res["D_iso"])
            D_FA_list.append(res["D_FA"])
            D_par_list.append(res["D_parallel"])
            D_perp_list.append(res["D_perp"])
            M2_list.append(res["M2"])

        # Average over all orientations (equivalent to MATLAB's mean(D_rot, 3))
        D_final      = D_stack.mean(axis=2)
        D_iso_final  = float(np.mean(D_iso_list))
        D_FA_final   = float(np.mean(D_FA_list))
        D_par_final  = float(np.mean(D_par_list))
        D_perp_final = float(np.mean(D_perp_list))
        M2_final     = float(np.mean(M2_list))
        orient_averaged = True

    # -------------------------------------------------------------------------
    # 5. Nearest-neighbour distances (Å) for reporting
    # -------------------------------------------------------------------------
    pts   = np.stack([x, y, z], axis=1)
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2)) * 1e10   # Å
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)

    # Sub-sample Cartesian coords for 3D plot (cap at 2000 for browser performance)
    plot_idx = np.random.choice(N, min(N, 2000), replace=False)

    return {
        "D":                    D_final.tolist(),
        "D_iso":                D_iso_final,
        "D_FA":                 D_FA_final,
        "D_parallel":           D_par_final,
        "D_perp":               D_perp_final,
        "M2":                   M2_final,
        "N_spins":              N,
        "concentration":        float(C_sim),
        "wigner_seitz_r":       float(wsr * 1e10),   # Å
        "orientation_averaged": orient_averaged,
        "num_orientations":     num_orientations if orient_averaged else 1,
        "nearest_neighbors": {
            "mean": float(nn.mean()),
            "min":  float(nn.min()),
            "max":  float(nn.max()),
        },
        "unit_cell": {k: float(v) for k, v in cell.items()},
        # Fractional coords of unique sites in one unit cell (for unit cell plot)
        "unique_coords": unique_coords.tolist(),
        # Subsampled Cartesian coords in nm (for spin cloud plot)
        "cartesian_sample": {
            "x": (x[plot_idx] * 1e9).tolist(),
            "y": (y[plot_idx] * 1e9).tolist(),
            "z": (z[plot_idx] * 1e9).tolist(),
        },
    }


# =============================================================================
#  CIF PARSING
# =============================================================================

def parse_cif(cif_text: str, nucleus: str, N_wanted: int, abund: float) -> tuple:
    """
    Parse a CIF file and return Cartesian coordinates (m) of the chosen nucleus.

    Returns
    -------
    coords : (N, 3) numpy array of Cartesian coordinates in metres
    cell   : dict with keys a, b, c (m), alpha, beta, gamma (rad)
    """
    lines = cif_text.splitlines()

    # --- Lattice parameters (strip uncertainty brackets like 5.46(2) → 5.46) ---
    def extract(pattern):
        m = re.search(pattern, cif_text)
        if not m:
            raise ValueError(f"Could not find '{pattern}' in CIF file.")
        return float(re.sub(r'\(\d+\)', '', m.group(1)))

    a     = 1e-10 * extract(r'_cell_length_a\s+([\d.]+(?:\(\d+\))?)')
    b     = 1e-10 * extract(r'_cell_length_b\s+([\d.]+(?:\(\d+\))?)')
    c     = 1e-10 * extract(r'_cell_length_c\s+([\d.]+(?:\(\d+\))?)')
    alpha = np.deg2rad(extract(r'_cell_angle_alpha\s+([\d.]+)'))
    beta  = np.deg2rad(extract(r'_cell_angle_beta\s+([\d.]+)'))
    gamma = np.deg2rad(extract(r'_cell_angle_gamma\s+([\d.]+)'))

    cell = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}

    # --- Find _atom_site block ---
    atom_start = next((i for i, l in enumerate(lines)
                       if l.strip().startswith("_atom_site_label")), None)
    if atom_start is None:
        raise ValueError("No _atom_site_label block found in CIF.")

    atom_lines = lines[atom_start:]

    # Find column index for fract_x (y and z follow immediately after)
    col_headers = [l.strip() for l in atom_lines if l.strip().startswith("_atom_site_")]
    fract_x_idx = next((i for i, h in enumerate(col_headers)
                        if "_atom_site_fract_x" in h), None)
    if fract_x_idx is None:
        raise ValueError("_atom_site_fract_x column not found in CIF.")

    # Find end of header block — first line that looks like data
    header_end = next((i for i, l in enumerate(atom_lines)
                       if not l.strip().startswith("_") and l.strip()
                       and not l.strip().startswith("#")
                       and len(l.strip()) > 5), len(atom_lines))

    # Parse atom rows
    raw_coords = []
    nucleus_pattern = re.compile(rf'^{re.escape(nucleus)}[\s\d(]|^{re.escape(nucleus)}$',
                                 re.IGNORECASE)

    for line in atom_lines[header_end:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("loop_") or line.startswith("_"):
            continue
        tokens = line.split()
        if len(tokens) < fract_x_idx + 3:
            continue
        label = tokens[0]
        # Match atoms whose label starts with the target nucleus letter
        if not re.match(rf'^{re.escape(nucleus)}[\d_\s(]|^{re.escape(nucleus)}$',
                        label, re.IGNORECASE):
            continue
        try:
            # Strip uncertainty parentheses from coordinate strings
            fx = float(re.sub(r'\(.*?\)', '', tokens[fract_x_idx]))
            fy = float(re.sub(r'\(.*?\)', '', tokens[fract_x_idx + 1]))
            fz = float(re.sub(r'\(.*?\)', '', tokens[fract_x_idx + 2]))
            raw_coords.append([fx, fy, fz])
        except (ValueError, IndexError):
            continue

    if not raw_coords:
        raise ValueError(f"No '{nucleus}' atom sites found in CIF file.")

    base_coords = np.array(raw_coords)

    # --- Apply symmetry operations ---
    unique_coords = apply_symmetry(lines, base_coords)

    # --- Replicate unit cell to reach N_wanted sites ---
    extended = replicate_cell(unique_coords, N_wanted)

    # --- Fractional → Cartesian via transformation matrix T ---
    volume_cell = (a * b * c
                   * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
                             + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)))
    T = np.array([
        [a,  b*np.cos(gamma),  c*np.cos(beta)],
        [0,  b*np.sin(gamma),  c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0,  0,                volume_cell / (a * b * np.sin(gamma))],
    ])
    cartesian = (T @ extended.T).T  # (N, 3) in metres

    # --- Isotopic abundance sub-sampling ---
    keep = np.random.rand(len(cartesian)) < abund
    cartesian = cartesian[keep]

    return cartesian, cell, unique_coords


def apply_symmetry(lines: list, base_coords: np.ndarray) -> np.ndarray:
    """
    Apply CIF symmetry operations to base_coords and return unique fractional coords.
    If no symmetry block is found, returns base_coords unchanged.
    """
    # Find symmetry operation block
    sym_idx = next((i for i, l in enumerate(lines)
                    if "_symmetry_equiv_pos_as_xyz" in l
                    or "_space_group_symop_operation_xyz" in l), None)

    if sym_idx is None:
        return np.unique(base_coords % 1.0, axis=0)

    # Parse symmetry operation strings
    sym_ops = []
    for line in lines[sym_idx + 1:]:
        line = re.sub(r'^\d+\s+', '', line.strip())  # Remove leading index
        line = line.replace("'", "")
        if not line or line.startswith("_") or line.startswith("loop_"):
            break
        sym_ops.append(line)

    all_coords = []
    for op in sym_ops:
        parts = [p.strip() for p in op.split(",")]
        if len(parts) != 3:
            continue
        for coord in base_coords:
            x_val, y_val, z_val = coord
            new = []
            for expr in parts:
                # Safely replace x/y/z with numerical values
                expr = re.sub(r'(?<![a-z])x(?![a-z])', str(x_val), expr, flags=re.IGNORECASE)
                expr = re.sub(r'(?<![a-z])y(?![a-z])', str(y_val), expr, flags=re.IGNORECASE)
                expr = re.sub(r'(?<![a-z])z(?![a-z])', str(z_val), expr, flags=re.IGNORECASE)
                try:
                    new.append(eval(expr))   # Safe here: expr contains only numbers and +-*/
                except Exception:
                    new.append(0.0)
            all_coords.append(new)

    if not all_coords:
        return np.unique(base_coords % 1.0, axis=0)

    all_coords = np.array(all_coords) % 1.0  # Fold into unit cell [0, 1)

    # Deduplicate with tolerance
    tol = 1e-4
    rounded = np.round(all_coords / tol) * tol
    unique  = np.unique(rounded, axis=0)
    return unique


def replicate_cell(unique_coords: np.ndarray, N_wanted: int) -> np.ndarray:
    """
    Tile the unit cell along a, b, c axes until we reach ≥ N_wanted sites.
    """
    n_base = len(unique_coords)
    Nrep   = max(0, int(np.ceil((N_wanted / n_base)**(1/3))) - 1)

    shifts = list(iproduct(range(Nrep + 1), range(Nrep + 1), range(Nrep + 1)))
    extended = []
    for coord in unique_coords:
        for da, db, dc in shifts:
            extended.append(coord + np.array([da, db, dc]))

    extended = np.array(extended)
    return np.unique(extended, axis=0)


# =============================================================================
#  POSITIONAL DISORDER
# =============================================================================

def apply_disorder(x, y, z, N, disorder, a, b, c, dist_min):
    """
    Randomly displace spin sites to simulate amorphous packing.
    Uses a hard-core exclusion of dist_min to prevent overlap.
    """
    pts = np.stack([x, y, z], axis=1).copy()

    for i in range(N):
        valid = False
        attempts = 0
        while not valid and attempts < 1000:
            displacement = disorder * np.array([a, b, c]) * (np.random.rand(3) - 0.5)
            proposed = pts[i] + displacement
            dists = np.linalg.norm(pts - proposed, axis=1)
            dists[i] = np.inf  # Ignore self
            valid = np.all(dists >= dist_min)
            attempts += 1
        if valid:
            pts[i] = proposed

    return pts[:, 0], pts[:, 1], pts[:, 2]


# =============================================================================
#  DIFFUSION TENSOR CALCULATION
# =============================================================================

def calculate_sdt(x, y, z, B0, gyro, N, omega_cs, dist_min):
    """
    Compute the 3×3 spin diffusion tensor D using the secular Redfield approach.

    Physics
    -------
    1. Pairwise dipolar coupling: B_ij = (μ₀/4π) ℏ γ² / r³ × (1 − 3cos²Θ)
    2. Second moment: M₂ = (1/N) Σ B_ij²
    3. ZQ spectral density: f_II = Gaussian(δν_ij, σ=√M₂)
    4. ZQ transition rate: W_II = (π/4) B_ij² f_II
    5. Diffusion tensor: D_αβ = (1/2N) Σ W_II × R_α × R_β   [nm² s⁻¹]

    Returns dict with D, D_iso, D_FA, D_parallel, D_perp, M2.
    """
    hbar = 1.05457182e-34   # J·s
    mu0  = 4 * np.pi * 1e-7  # T·m·A⁻¹

    # --- Pairwise displacement matrices (N×N) ---
    Rx = x[:, None] - x[None, :]   # equivalently: X2 - X1 in MATLAB meshgrid convention
    Ry = y[:, None] - y[None, :]
    Rz = z[:, None] - z[None, :]

    # --- Minimum-image periodic boundary conditions ---
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    Lz = z.max() - z.min()
    Rx -= np.round(Rx / Lx) * Lx
    Ry -= np.round(Ry / Ly) * Ly
    Rz -= np.round(Rz / Lz) * Lz

    # --- Pairwise distances (self = NaN) ---
    Distance = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    Distance[Distance < dist_min] = np.nan  # Exclude self-pairs & overlapping sites

    # --- Angle Θ between inter-spin vector and B₀ ---
    dot    = Rx * B0[0] + Ry * B0[1] + Rz * B0[2]
    cosTheta = dot / Distance            # NaN where Distance is NaN
    Theta  = np.arccos(np.clip(cosTheta, -1, 1))

    # --- Dipolar coupling constant b_ij (Hz) ---
    bij = (mu0 / (4 * np.pi)) * (gyro**2 * hbar) / Distance**3
    bij = np.nan_to_num(bij, nan=0.0, posinf=0.0, neginf=0.0)

    # Secular (orientation-dependent) part B_ij
    Bij         = bij * (1 - 3 * np.cos(Theta)**2)
    Bij         = np.nan_to_num(Bij, nan=0.0)
    Bij_squared = Bij**2

    # --- Second moment M₂ ---
    M2 = Bij_squared.sum() / N

    # --- Zero-quantum spectral density f_II (Gaussian lineshape) ---
    # δν_ij = ½(ω_i − ω_j), f_II centred at 0 with width √M₂
    nu_diff = 0.5 * (omega_cs[:, None] - omega_cs[None, :])  # N×N
    if M2 > 0:
        fII = (1 / np.sqrt(2 * np.pi * M2)) * np.exp(-nu_diff**2 / (2 * M2))
    else:
        fII = np.zeros_like(nu_diff)

    # --- Zero-quantum transition rate W_II ---
    WII = (np.sqrt(np.pi) / 4) * Bij_squared * fII

    # --- Build the 3×3 diffusion tensor ---
    # Factor 1e18 converts m² → nm²
    normD  = 1e18 / (2 * N)
    D      = np.zeros((3, 3))
    comps  = [Rx, Ry, Rz]

    for i in range(3):
        for j in range(i, 3):
            D[i, j] = normD * (WII * comps[i] * comps[j]).sum()
            D[j, i] = D[i, j]   # Symmetrise

    D = np.real(D)

    # --- Derived scalar quantities ---
    eigenvalues  = np.sort(np.linalg.eigvalsh(D))
    D_iso        = np.trace(D) / 3
    delta_D      = eigenvalues - D_iso
    denom        = np.sqrt((eigenvalues**2).sum())
    D_FA         = (np.sqrt(1.5) * np.linalg.norm(delta_D) / denom) if denom > 0 else 0.0
    D_parallel   = float(B0 @ D @ B0)
    D_perp       = (np.trace(D) - D_parallel) / 2

    return {
        "D":           D,
        "D_iso":       D_iso,
        "D_FA":        D_FA,
        "D_parallel":  D_parallel,
        "D_perp":      D_perp,
        "M2":          M2,
        "eigenvalues": eigenvalues.tolist(),
    }
