"""
sdt_calc.py  —  Spin Diffusion Tensor Calculator
Physics: secular Redfield theory, MAS rotor-harmonic expansion, powder averaging.
"""

import re
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

# =============================================================================
#  PUBLIC ENTRY POINT  —  generator yielding progress strings then result dict
# =============================================================================

def run_calculation(cif_text, nucleus="F", N_wanted=1000, B0_field=9.4,
                    disorder=0.0, dist_min=2e-10, num_orientations=50,
                    mas_rate_khz=0.0, abund_mode="natural", abund_pct=None):
    """
    Generator. Yields "progress:N:TOTAL" strings during orientation loop,
    then finally yields the result dict.
    """
    if nucleus not in NUCLEUS_PARAMS:
        raise ValueError(f"Nucleus '{nucleus}' not recognised. Choose: {list(NUCLEUS_PARAMS)}")

    gyro = NUCLEUS_PARAMS[nucleus]["gyro"]

    # Isotopic abundance
    if abund_mode == "natural":
        abund = NUCLEUS_PARAMS[nucleus]["abund"]
    else:
        if abund_pct is None:
            raise ValueError("abund_pct required when abund_mode='custom'")
        abund = max(0.0, min(1.0, float(abund_pct) / 100.0))

    omega_r = mas_rate_khz * 1e3   # kHz → Hz
    Na = 6.02214076e23
    B0 = np.array([0.0, 0.0, 1.0])

    # 1. Parse CIF
    coords, cell, unique_coords = parse_cif(cif_text, nucleus, N_wanted, abund)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    N = len(x)
    if N < 2:
        raise ValueError("Fewer than 2 spin sites found. Check CIF and nucleus.")

    # 2. Disorder
    if disorder > 0:
        x, y, z = apply_disorder(x, y, z, N, disorder,
                                 cell["a"], cell["b"], cell["c"], dist_min)

    omega_cs = np.zeros(N)

    # 3. Concentration / Wigner-Seitz
    a, b, c = cell["a"], cell["b"], cell["c"]
    al, be, ga = cell["alpha"], cell["beta"], cell["gamma"]
    Lx, Ly, Lz = x.max()-x.min(), y.max()-y.min(), z.max()-z.min()
    volume = ((Lx+a/2)*(Ly+b/2)*(Lz+c/2)
              * np.sqrt(1-np.cos(al)**2-np.cos(be)**2-np.cos(ga)**2
                        +2*np.cos(al)*np.cos(be)*np.cos(ga)))
    C_sim = N / (volume * 1e3 * Na)
    wsr   = 0.1 * (3/(4*np.pi*C_sim*Na))**(1/3)

    # 4. Tensor calculation
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

        D_stack = np.zeros((3,3,num_orientations))
        D_iso_l, D_FA_l, D_par_l, D_perp_l, M2_l = [], [], [], [], []

        for o in range(num_orientations):
            ca, sa = np.cos(alpha_rot[o]), np.sin(alpha_rot[o])
            cb, sb = np.cos(beta_rot[o]),  np.sin(beta_rot[o])
            Rot = np.array([[cb*ca, -sa, sb*ca],
                            [cb*sa,  ca, sb*sa],
                            [-sb,     0,    cb]])
            B0rot = Rot @ B0
            res = calculate_sdt(x, y, z, B0rot, gyro, N, omega_cs, dist_min, omega_r)
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

        # Diagonalise the averaged D for PAS
        eig_vals_f, eig_vecs_tmp = np.linalg.eigh(D_final)
        idx = np.argsort(eig_vals_f)
        eig_vals_f = eig_vals_f[idx]
        eig_vecs_f = eig_vecs_tmp[:, idx].T   # rows = eigenvectors

    # PAS display values
    pas_eigenvalues  = eig_vals_f.tolist() if hasattr(eig_vals_f,'tolist') else list(eig_vals_f)
    pas_eigenvectors = eig_vecs_f.tolist() if hasattr(eig_vecs_f,'tolist') else [list(r) for r in eig_vecs_f]

    # 5. Nearest neighbours
    pts   = np.stack([x,y,z], axis=1)
    diffs = pts[:,None,:]-pts[None,:,:]
    dists = np.sqrt((diffs**2).sum(axis=2))*1e10
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)

    plot_idx = np.random.choice(N, min(N,2000), replace=False)

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
        "PAS": {
            "eigenvalues":  pas_eigenvalues,
            "eigenvectors": pas_eigenvectors,
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
#  DIFFUSION TENSOR  (static + MAS)
# =============================================================================

def calculate_sdt(x, y, z, B0, gyro, N, omega_cs, dist_min, omega_r=0.0):
    hbar = 1.05457182e-34
    mu0  = 4*np.pi*1e-7

    Rx = x[:,None]-x[None,:]; Ry = y[:,None]-y[None,:]; Rz = z[:,None]-z[None,:]
    Lx,Ly,Lz = x.max()-x.min(), y.max()-y.min(), z.max()-z.min()
    Rx -= np.round(Rx/Lx)*Lx; Ry -= np.round(Ry/Ly)*Ly; Rz -= np.round(Rz/Lz)*Lz

    Distance = np.sqrt(Rx**2+Ry**2+Rz**2)
    Distance[Distance < dist_min] = np.nan

    dot      = Rx*B0[0]+Ry*B0[1]+Rz*B0[2]
    cosTheta = dot/Distance
    Theta    = np.arccos(np.clip(cosTheta,-1,1))

    bij         = (mu0/(4*np.pi))*(gyro**2*hbar)/Distance**3
    bij         = np.nan_to_num(bij)
    Bij         = bij*(1-3*np.cos(Theta)**2)
    Bij         = np.nan_to_num(Bij)
    Bij_squared = Bij**2
    M2          = Bij_squared.sum()/N

    nu_diff = 0.5*(omega_cs[:,None]-omega_cs[None,:])

    def J(delta):
        if M2 > 0:
            return (1/np.sqrt(2*np.pi*M2))*np.exp(-delta**2/(2*M2))
        return np.zeros_like(delta)

    # Magic angle beta_m = arccos(1/sqrt(3)) — always used for rotor geometry
    beta_m = np.arccos(1/np.sqrt(3))
    ux, uy, uz = np.sin(beta_m), 0.0, np.cos(beta_m)

    # Rotor-phase averaged spatial weights T_αβ
    r2     = Rx**2+Ry**2+Rz**2
    udotr  = ux*Rx+uy*Ry+uz*Rz
    rpar2  = udotr**2; rperp2 = r2-rpar2
    Txx = 0.5*rperp2*(1-ux*ux)+rpar2*ux*ux
    Tyy = 0.5*rperp2*(1-uy*uy)+rpar2*uy*uy
    Tzz = 0.5*rperp2*(1-uz*uz)+rpar2*uz*uz
    Txy = 0.5*rperp2*(-ux*uy)+rpar2*ux*uy
    Txz = 0.5*rperp2*(-ux*uz)+rpar2*ux*uz
    Tyz = 0.5*rperp2*(-uy*uz)+rpar2*uy*uz

    # Wigner factors at magic angle
    f0sq  = ((3*np.cos(beta_m)**2-1)/2)**2
    f1sq  = (1.5*np.sin(beta_m)*np.cos(beta_m))**2
    f2sq  = (0.75*np.sin(beta_m)**2)**2
    Sbeta = f0sq+2*f1sq+2*f2sq

    mList  = [-2,-1, 0, 1, 2]
    fSq    = [f2sq, f1sq, f0sq, f1sq, f2sq]

    WII = np.zeros_like(Bij_squared)
    for m, f2m in zip(mList, fSq):
        Delta_m = nu_diff - m*omega_r
        WII += (np.sqrt(np.pi)/4)*Bij_squared*f2m*J(Delta_m)/Sbeta

    normD = 1e18/(2*N)
    T_comps = [[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]]
    D = np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            D[i,j] = normD*(WII*T_comps[i][j]).sum()
            D[j,i] = D[i,j]
    D = np.real(D)

    eig_vals, eig_vecs = np.linalg.eigh(D)
    idx = np.argsort(eig_vals)
    eig_vals = eig_vals[idx]; eig_vecs = eig_vecs[:,idx]

    D_iso      = np.trace(D)/3
    delta_D    = eig_vals-D_iso
    denom      = np.sqrt((eig_vals**2).sum())
    D_FA       = (np.sqrt(1.5)*np.linalg.norm(delta_D)/denom) if denom>0 else 0.0
    D_parallel = float(B0@D@B0)
    D_perp     = (np.trace(D)-D_parallel)/2

    return {
        "D": D, "D_iso": D_iso, "D_FA": D_FA,
        "D_parallel": D_parallel, "D_perp": D_perp, "M2": M2,
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
        if not m:
            raise ValueError(f"Pattern not found in CIF: {pattern}")
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

    # Isotope sub-sampling
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
