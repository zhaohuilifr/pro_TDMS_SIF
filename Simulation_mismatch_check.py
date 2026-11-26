import glob, os, numpy as np, pandas as pd
root = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new8\Res_67_761_LAIlt_full_de_Cabdiff_newNPQparams'
files = sorted(glob.glob(os.path.join(root, 'SIF_canopy', '*_2023_*_canopy.csv')))[::50]   # 抽样约 1/50
rows = []
for f in files:
    fl = os.path.join(root, 'SIF_layers', os.path.basename(f).replace('canopy', 'lys_ba'))
    if not os.path.exists(fl):
        continue
    c = pd.read_csv(f, index_col=False)
    l = pd.read_csv(fl, index_col=False)
    par_rad = c.loc[(c.wl >= 400) & (c.wl <= 700), 'irradiance'].sum()   # Radiance 用的当前 PAR（谱积分）
    rows.append({'file': os.path.basename(f), 'PAR_Radiance': par_rad,
                 'PARinLAI_top': l['PARinLAI'].iloc[0]})                # boucle_lai 写入的 PAR（第 1 层）
d = pd.DataFrame(rows)
d['ratio'] = d.PARinLAI_top / d.PAR_Radiance
print(d.ratio.describe())
print(d.sort_values('ratio').iloc[[0, 1, 2, -3, -2, -1]])