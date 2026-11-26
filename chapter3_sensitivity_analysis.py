"""
sa_sif.py - sampling, parallel running and analysis for the CASTANEA-SIF sensitivity analysis
(case_simul = 40, see PATCHES.md).

Run from the CASTANEA root folder (the one containing ini/, data/, job.txt), e.g.
    python sa_sif.py oat    --ini Barbeau_SA.ini --nproc 8
    python sa_sif.py morris --ini Barbeau_SA.ini --r 30 --nproc 8
    python sa_sif.py sobol  --ini Barbeau_SA.ini --n 1024 --params kDF alpha_ql half_satuPAR Cab LMAX NC2 rdtq agreg
    python sa_sif.py analyse --design morris --result result_SA/
Requirements: numpy, pandas, SALib (pip install SALib), matplotlib for the figures.
"""
import argparse, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# 1. Parameter table: name -> (base, low, high, group)
#    !!! base values: copy the ones actually used in your Chapter-3 run;
#        ranges: justify each one in Table 4.1 (literature / measurements at Barbeau).
# ------------------------------------------------------------------------------------
PARAMS = {
    # fluorescence (FluoLAI / FluorMODleaf)
    "kDF":          (6.5,   3.0,   12.0,  "fluo"),   # kD/kF ratio
    "phiPSIImax":   (0.81,  0.76,  0.85,  "fluo"),   # Fv/Fm of dark-adapted leaves
    "alpha_ql":     (0.67,  0.40,  1.00,  "fluo"),   # qL = alpha*exp(-PAR/beta)
    "half_satuPAR": (761.7, 300.0, 1500., "fluo"),   # PAR at which qL = alpha/2
    "ps2LT":        (0.50,  0.35,  0.65,  "fluo"),   # FluorMODleaf PSII lifetime (check unit)
    "ps1LT":        (0.035, 0.025, 0.045, "fluo"),   # FluorMODleaf PSI lifetime
    # leaf biochemistry / optics (PROSPECT-D + FluorMODleaf); multipliers of leaf_parameter.txt
    "Cab_x":        (1.0,   0.6,   1.4,   "leaf"),
    "Car_x":        (1.0,   0.6,   1.4,   "leaf"),
    "Nleaf_x":      (1.0,   0.8,   1.3,   "leaf"),
    "Cm_x":         (1.0,   0.7,   1.3,   "leaf"),
    "Cw_x":         (1.0,   0.7,   1.3,   "leaf"),
    # canopy structure
    "LMAX":         (6.0,   4.0,   8.0,   "canopy"), # max LAI (forced)
    "agreg":        (1.0,   0.6,   1.0,   "canopy"), # clumping (agreginit)
    "alphal_deg":   (24.0,  15.0,  60.0,  "canopy"), # mean leaf inclination
    # photosynthesis / stomata (CASTANEA)
    "NC2":          (20.0,  14.0,  32.0,  "photo"),  # Vcmax-N slope
    "coefbeta":     (2.1,   1.6,   2.6,   "photo"),  # Jmax/Vcmax
    "rdtq":         (0.292, 0.22,  0.35,  "photo"),  # quantum yield of electron transport
    "teta":         (0.7,   0.5,   0.95,  "photo"),  # curvature
    "g1max":        (9.5,   6.0,   14.0,  "photo"),  # stomatal slope
}

OUTPUTS = ["SIF760_md", "SIF687_md", "SIFem760_md", "fesc760_md", "SIFPSII_md",
           "GPP_md", "APAR_md", "GPP_gs_sum", "SIF760_over_APAR", "GPP_over_SIF760",
           "slope_GPP_SIF760", "R2_GPP_SIF760"]


def problem(names):
    return {"num_vars": len(names), "names": list(names),
            "bounds": [[PARAMS[n][1], PARAMS[n][2]] for n in names]}


# ------------------------------------------------------------------------------------
# 2. Designs
# ------------------------------------------------------------------------------------
def design_oat(names, rel=(-0.2, -0.1, 0.1, 0.2)):
    """Local one-at-a-time: base + base*(1+d) for each parameter (clipped to bounds)."""
    base = {n: PARAMS[n][0] for n in names}
    rows = [dict(base, run_id="base")]
    for n in names:
        for d in rel:
            r = dict(base)
            r[n] = float(np.clip(base[n] * (1 + d), PARAMS[n][1], PARAMS[n][2]))
            r["run_id"] = f"{n}_{int(round(d*100)):+d}"
            rows.append(r)
    return pd.DataFrame(rows)[["run_id"] + list(names)]


def design_morris(names, r=30, levels=4, seed=1):
    from SALib.sample import morris as ms
    X = ms.sample(problem(names), N=r, num_levels=levels, seed=seed)
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "run_id", [f"m{i:05d}" for i in range(len(df))])
    return df


def design_sobol(names, n=1024, seed=1):
    from SALib.sample import sobol as ss   # Saltelli scheme, N*(k+2) runs without 2nd order
    X = ss.sample(problem(names), N=n, calc_second_order=False, seed=seed)
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "run_id", [f"s{i:06d}" for i in range(len(df))])
    return df


# ------------------------------------------------------------------------------------
# 3. Write chunks + ini files, run in parallel
# ------------------------------------------------------------------------------------
def write_chunks(df, ini_template, nproc, root=Path("."), tag="SA"):
    """Split the design into nproc chunks: SA/<tag>_cNN_samples.csv + ini/<tag>_cNN.ini."""
    (root / "SA").mkdir(exist_ok=True)
    ini_dir = root / "ini"
    txt = (ini_dir / ini_template).read_text(encoding="latin-1")
    # force case_simul = 40 on the line containing '!case_simul'
    txt, nsub = re.subn(r"^\s*\d+(\s*!case_simul)", r"40\1", txt, flags=re.M)
    if nsub != 1:
        sys.exit("could not find the '!case_simul' line in the ini template")
    df.to_csv(root / "SA" / f"{tag}_design_all.csv", index=False)
    inis = []
    for c, idx in enumerate(np.array_split(np.arange(len(df)), nproc)):
        part = df.iloc[idx]
        name = f"{tag}_c{c:02d}"
        part.to_csv(root / "SA" / f"{name}_samples.csv", index=False)
        (ini_dir / f"{name}.ini").write_text(txt, encoding="latin-1")
        inis.append(f"{name}.ini")
    return inis


def run_parallel(exe, inis, workdir):
    """Launch one castanea process per ini (needs the lec_job patch, PATCHES.md section 7)."""
    t0 = time.time()
    procs = [subprocess.Popen([exe, ini], cwd=workdir,
                              stdout=open(Path(workdir) / f"{ini}.log", "w"),
                              stderr=subprocess.STDOUT) for ini in inis]
    for p in procs:
        p.wait()
    print(f"{len(inis)} processes finished in {(time.time()-t0)/60:.1f} min")


def collect(result_dir, tag="SA"):
    files = sorted(Path(result_dir).glob(f"{tag}_c*_SA_summary.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.replace(-999, np.nan)
    df.to_csv(Path(result_dir) / f"{tag}_summary_all.csv", index=False)
    return df


# ------------------------------------------------------------------------------------
# 4. Analysis
# ------------------------------------------------------------------------------------
def analyse_oat(df, names):
    base = df[df.run_id == "base"].iloc[0]
    out = []
    for n in names:
        sub = df[df.run_id.str.startswith(n + "_")]
        for y in OUTPUTS:
            # normalised sensitivity (elasticity) S = (dY/Y)/(dX/X), averaged over the +-10 % runs
            s = [((r[y] - base[y]) / base[y]) / ((r[n] - base[n]) / base[n])
                 for _, r in sub.iterrows() if abs(r[n] - base[n]) > 0 and base[y] != 0]
            out.append({"param": n, "output": y, "elasticity": np.nanmean(s) if s else np.nan})
    return pd.DataFrame(out).pivot(index="param", columns="output", values="elasticity")


def analyse_morris(design, res, names, levels=4):
    from SALib.analyze import morris as ma
    X = design[names].values
    out = {}
    for y in OUTPUTS:
        Y = res.set_index("run_id").loc[design.run_id, y].values
        if np.isnan(Y).any():
            print(f"{y}: {np.isnan(Y).sum()} NaN -> skipped"); continue
        si = ma.analyze(problem(names), X, Y, num_levels=levels, print_to_console=False)
        out[y] = pd.DataFrame({"mu_star": si["mu_star"], "sigma": si["sigma"],
                               "mu_star_conf": si["mu_star_conf"]}, index=names)
    return out


def analyse_sobol(design, res, names):
    from SALib.analyze import sobol as sa
    out = {}
    for y in OUTPUTS:
        Y = res.set_index("run_id").loc[design.run_id, y].values
        if np.isnan(Y).any():
            print(f"{y}: {np.isnan(Y).sum()} NaN -> skipped"); continue
        si = sa.analyze(problem(names), Y, calc_second_order=False, print_to_console=False)
        out[y] = pd.DataFrame({"S1": si["S1"], "S1_conf": si["S1_conf"],
                               "ST": si["ST"], "ST_conf": si["ST_conf"]}, index=names)
    return out


def sobol_daily(design, result_dir, names, var="SIF760_md", tag="SA"):
    """Time-varying first-order/total indices from the daily files (needs sa_write_daily)."""
    from SALib.analyze import sobol as sa
    daily = {}
    for rid in design.run_id:
        f = next(Path(result_dir, "SA_daily").glob(f"{tag}_c*_{rid}.csv"))
        d = pd.read_csv(f).replace(-999, np.nan)
        daily[rid] = d.set_index(["year", "doy"])[var]
    M = pd.DataFrame(daily)        # rows = days, columns = runs (same order as design)
    rows = []
    for (yr, doy), Y in M.iterrows():
        Y = Y[design.run_id].values
        if np.isnan(Y).any() or np.std(Y) == 0:
            continue
        si = sa.analyze(problem(names), Y, calc_second_order=False, print_to_console=False)
        rows += [{"year": yr, "doy": doy, "param": n, "S1": si["S1"][i], "ST": si["ST"][i]}
                 for i, n in enumerate(names)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oat", "morris", "sobol", "analyse"])
    ap.add_argument("--ini", help="ini template in ini/ (must contain a '!case_simul' line)")
    ap.add_argument("--exe", default="castanea.exe")
    ap.add_argument("--workdir", default="projet_fortran_ifx_Castanea",
                    help="folder the exe is launched from (its parent must hold job.txt, ini/, SA/)")
    ap.add_argument("--nproc", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--params", nargs="*", default=list(PARAMS))
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--design", choices=["oat", "morris", "sobol"])
    ap.add_argument("--result", default="result_SA")
    ap.add_argument("--dry", action="store_true", help="only write the design, do not run")
    a = ap.parse_args()
    names = a.params

    if a.mode != "analyse":
        df = {"oat": lambda: design_oat(names),
              "morris": lambda: design_morris(names, r=a.r),
              "sobol": lambda: design_sobol(names, n=a.n)}[a.mode]()
        print(f"{a.mode}: {len(df)} runs, {len(names)} parameters")
        inis = write_chunks(df, a.ini, a.nproc, tag=a.mode)
        if not a.dry:
            run_parallel(a.exe, inis, a.workdir)
        return

    tag = a.design
    design = pd.read_csv(Path("SA") / f"{tag}_design_all.csv")
    names = [c for c in design.columns if c != "run_id"]
    res = collect(a.result, tag=tag)
    if tag == "oat":
        print(analyse_oat(res, names).round(3).to_string())
    elif tag == "morris":
        for y, t in analyse_morris(design, res, names).items():
            print(f"\n== {y}\n", t.sort_values("mu_star", ascending=False).round(4).to_string())
    else:
        for y, t in analyse_sobol(design, res, names).items():
            print(f"\n== {y}\n", t.sort_values("ST", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()