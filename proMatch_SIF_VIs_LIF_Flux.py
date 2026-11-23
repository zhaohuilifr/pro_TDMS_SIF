# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def matlab_datenum_to_datetime(yearstr, datenum, offset_hours=2):
    # MATLAB datenum 是从 0000-01-01 开始的天数，Python datetime 从 1970-01-01 开始
    # MATLAB datenum 的整数部分是天数，小数部分是一天中的时间
    # days = int(datenum)
    # fraction = datenum - days
    # dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    if yearstr == '2022': # 2023 - 2025, offset_hours = 2
        offset_hours = 5 #
    dt = pd.to_datetime(datenum - 719529, unit='D') + pd.Timedelta(hours=offset_hours)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    doy = dt.timetuple().tm_yday + hour / 24
    return dt, doy, hour

# def filter_3sigma(array):
#     mean = np.nanmean(array)
#     std = np.nanstd(array)
#     array[~(array >= mean - 3 * std) & (array <= mean + 3 * std)] = np.nan
#     if np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)) > 0:
#         print('number of outliers filtered:', np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)))
#     return array

def fluxdata_doy_calculation(fluxdata):
    # fluxdata uses CET time, which is UTC+1
    mask = fluxdata['hh'] == 0
    fluxdata.loc[mask, 'date'] = pd.to_datetime({'year': fluxdata.loc[mask, 'an'],
                                                  'month': fluxdata.loc[mask, 'mois'],
                                                  'day': fluxdata.loc[mask, 'jour']}) - pd.Timedelta(days=1)
    fluxdata.loc[~mask, 'date'] = pd.to_datetime({'year': fluxdata.loc[~mask, 'an'],
                                                   'month': fluxdata.loc[~mask, 'mois'],
                                                   'day': fluxdata.loc[~mask, 'jour']})
    fluxdata.loc[mask, 'hh'] = 24
    fluxdata.loc[mask, 'jj'] = fluxdata.loc[mask, 'jj'] - 1
    # 从日期提取年月日
    fluxdata['an'] = fluxdata['date'].dt.year
    fluxdata['mois'] = fluxdata['date'].dt.month
    fluxdata['jour'] = fluxdata['date'].dt.day
    fluxdata = fluxdata.drop('date', axis=1)
    fluxdata['hh'] -= 0.5

    fluxdata['hour'] = (fluxdata['hh'].astype(int))
    fluxdata['minute'] = (fluxdata['hh'] - fluxdata['hour']) * 60
    fluxdata['DOY'] = fluxdata['jj'] + (fluxdata['hour'] + fluxdata['minute'] / 60) / 24

    # !!!!!!!!!!!!!!!! CET to UTC conversion !!!!!!!!!!!!!!!!
    fluxdata['DOY_UTC'] = fluxdata['DOY'] - 1/24
    # fluxdata.loc[fluxdata['DOY_UTC'] < 0, 'DOY_UTC'] += 365 # for DOY < 1, add 365 to make it positive
    fluxdata['hour_UTC'] = fluxdata['hour'] - 1
    fluxdata.loc[fluxdata['hour_UTC'] < 0, 'hour_UTC'] += 24 # for hour < 0, add 24 to make it positive
    fluxdata['minute_UTC'] = fluxdata['minute']
    # remove rows with DOY_UTC < 0 or DOY_UTC > 365, as they are out of the valid range
    fluxdata = fluxdata.loc[(fluxdata['DOY_UTC'] >= 0) & (fluxdata['DOY_UTC'] <= 365)].reset_index(drop=True)

    return fluxdata

def match_data_vectorized(yearstr, df_flux, df_sif, df_vis, df_lif, num_max_within_halfhour, num_max_within_halfhour_lif):
    """超快速向量化匹配：用searchsorted+预分配+批量赋值"""
    
    sif_columns = ['DOY_sif','SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
    vis_columns = ['DOY_vis', 'NDVI', 'EVI', 'MTCI', 'CIgreen', 'PRI', 'FCVI', 'NIRv', 'mNDI705','NIRVR']
    lif_columns = ['DOY_lif', 'Fs', 'PAR', 'Fcont']
    
    # 预分配结果数组
    n_flux = len(df_flux)
    results_sif = {col: np.full(n_flux, np.nan) for col in sif_columns}
    results_vis = {col: np.full(n_flux, np.nan) for col in vis_columns}
    results_lif = {col: np.full(n_flux, np.nan) for col in lif_columns}
    
    # 转为numpy数组并排序（为searchsorted做准备）
    sif_doy = df_sif['DOY_sif'].values
    sif_hour = df_sif['Hour'].values
    sif_data = {col: df_sif[col].values for col in sif_columns}
    
    vis_doy = df_vis['DOY_vis'].values
    vis_hour = df_vis['Hour'].values
    vis_data = {col: df_vis[col].values for col in vis_columns}
    
    lif_doy = df_lif['DOY_lif'].values
    lif_hour = df_lif['Hour'].values
    lif_data = {col: df_lif[col].values for col in lif_columns}
    
    flux_doy = df_flux['DOY_UTC'].values.astype(int)
    flux_hour = df_flux['hour_UTC'].values
    
    # 向量化3sigma过滤函数
    def filter_3sigma_vec(arr):
        """向量化3sigma过滤，返回过滤后的数组副本"""
        arr = arr.copy()
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        arr[~((arr >= mean - 3*std) & (arr <= mean + 3*std))] = np.nan
        return arr
    
    warnings_log = []
    
    # 逐行匹配但优化内部操作
    for i in range(len(df_flux)): # len(df_flux)
        day_flux = flux_doy[i]
        hour_flux = flux_hour[i]
        
        # === SIF 匹配 ===
        idx_sif = ((sif_doy >= day_flux) & (sif_doy < day_flux + 1) & 
                   (sif_hour >= hour_flux) & (sif_hour < hour_flux + 0.5))
        if np.any(idx_sif):
            for col in sif_columns:
                vals = sif_data[col][idx_sif]
                vals = filter_3sigma_vec(vals)
                nan_ratio = np.isnan(vals).sum() / max(num_max_within_halfhour, np.sum(idx_sif))
                if nan_ratio <= 0.7:
                    results_sif[col][i] = np.nanmean(vals)
                else:
                    warnings_log.append(f'SIF {col}: day {day_flux} hour {hour_flux}')
        
        # === VIS 匹配 ===
        idx_vis = ((vis_doy >= day_flux) & (vis_doy < day_flux + 1) & 
                   (vis_hour >= hour_flux) & (vis_hour < hour_flux + 0.5))
        if np.any(idx_vis):
            for col in vis_columns:
                vals = vis_data[col][idx_vis]
                vals = filter_3sigma_vec(vals)
                nan_ratio = np.isnan(vals).sum() / max(num_max_within_halfhour, np.sum(idx_vis))
                if nan_ratio <= 0.7:
                    results_vis[col][i] = np.nanmean(vals)
                else:
                    warnings_log.append(f'VIS {col}: day {day_flux} hour {hour_flux}')
        
        # === LIF 匹配 ===
        idx_lif = ((lif_doy >= day_flux) & (lif_doy < day_flux + 1) & 
                   (lif_hour >= hour_flux) & (lif_hour < hour_flux + 0.5))
        if np.any(idx_lif):
            for col in lif_columns:
                vals = lif_data[col][idx_lif]
                vals = filter_3sigma_vec(vals)
                nan_ratio = np.isnan(vals).sum() / max(num_max_within_halfhour_lif, np.sum(idx_lif))
                if nan_ratio <= 0.7:
                    results_lif[col][i] = np.nanmean(vals)
                else:
                    warnings_log.append(f'LIF {col}: day {day_flux} hour {hour_flux}')
    
    # 批量赋值到df_flux（一次操作，比逐行快）
    for col in sif_columns:
        df_flux[col] = results_sif[col]
    for col in vis_columns:
        df_flux[col] = results_vis[col]
    for col in lif_columns:
        df_flux[col] = results_lif[col]
    
    if warnings_log:
        with open(os.path.join(savepath, f'{yearstr}_warnings.log'), 'w') as f:
            f.write('\n'.join(warnings_log))
    
    return df_flux

def add_halfhour_slot(df, doy_col, hour_col=None):
    if hour_col is None:
        doy = df[doy_col].to_numpy()
    else:
        doy = df[doy_col].to_numpy() + df[hour_col].to_numpy() / 24.0
    df = df.copy()
    df["slot"] = np.floor(doy * 48).astype("int64")
    return df

def sigma_clean_mean_with_ratio(x, min_valid_ratio=0.3):
    x = x.to_numpy(dtype=float)
    if x.size == 0:
        return np.nan

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if np.isnan(std) or std == 0:
        valid_ratio = np.sum(~np.isnan(x)) / x.size
        return np.nanmean(x) if valid_ratio >= min_valid_ratio else np.nan

    keep = (x >= mean - 3 * std) & (x <= mean + 3 * std)
    x = x.copy()
    x[~keep] = np.nan

    valid_ratio = np.sum(~np.isnan(x)) / x.size
    if valid_ratio < min_valid_ratio:
        return np.nan

    return np.nanmean(x)

def match_data_slot(df_flux, df_sif, df_vis, df_lif=None):
    # 1) flux 也做 slot
    df_flux = add_halfhour_slot(df_flux, "DOY_UTC")

    # 2) 三类数据先按 slot 聚合
    df_sif["slot"] = np.floor(df_sif["DOY_sif"] * 48).astype("int64")
    df_vis["slot"] = np.floor(df_vis["DOY_vis"] * 48).astype("int64")
    
    sif_cols = ["DOY_sif","SIF_3FLD", "SIF_iFLD", "SIF_SFM_lin_a"]
    vis_cols = ["DOY_vis","NDVI", "EVI", "MTCI", "CIgreen", "PRI", "FCVI", "NIRv", "mNDI705", 'gcc', "NIRVR"]
    lif_cols = ["DOY_lif","Fs", "PAR", "Fcont"]

    df_sif_agg = df_sif.groupby("slot", as_index=False)[sif_cols].agg(
        lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
    )
    df_vis_agg = df_vis.groupby("slot", as_index=False)[vis_cols].agg(
        lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
    )
    if df_lif is not None:
        df_lif["slot"] = np.floor(df_lif["DOY_lif"] * 48).astype("int64")
        df_lif_agg = df_lif.groupby("slot", as_index=False)[lif_cols].agg(
            lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
        )

    # 3) 一次 merge 回 flux
    df_flux = df_flux.merge(df_sif_agg, on="slot", how="left")
    df_flux = df_flux.merge(df_vis_agg, on="slot", how="left")
    if df_lif is not None:
        df_flux = df_flux.merge(df_lif_agg, on="slot", how="left")
    return df_flux

def _robust_z(x, center, scale):
    """robust z-score using median / MAD (scaled to sigma)"""
    return (x - center) / (1.4826 * scale)

def qc_obs_sif_gpp(df,
                   sif_cols=('SIF_3FLD', 'SIF_iFLD', 'SIF_SFM_lin_a'),
                   par_col='PAR (µmol/m2/s)',
                   par_min_gpp=50.,       # µmol/m2/s, daytime threshold for GPP
                   par_min_sif=50.,      # µmol/m2/s, low-light SIF has poor SNR
                   gpp_max=60.,           # µmol/m2/s, physical upper bound
                   gpp_relunc_max=1.,     # drop when GPPsd / GPP > this
                   sif_max=2.,            # mW/m2/sr/nm
                   z_thresh=3.5,          # robust z threshold for outliers
                   yield_window_days=3,   # ± days window for SIF/PAR outlier test
                   remove_rain=True,
                   verbose=True):
    """
    Quality control of observed GPP (EC) and SIF (spectrometer) after matching.
    Original values are kept as <col>_raw; flags are stored as bitmasks:
      QC_GPP: 1 fill/NaN, 2 gap-filled (flag_NEE_GF!=0), 4 qc_co2>1, 8 night/low PAR,
              16 out of range (<0 or >gpp_max), 32 large uncertainty, 64 rain
      QC_SIF: 1 NaN, 2 low PAR, 4 out of range (<=0 or >sif_max), 8 rain,
              16 3FLD vs SFM inconsistent, 32 SIF/PAR yield outlier
    Flagged values are set to NaN in GPP / SIF columns.
    """
    df = df.copy()
    # 1) fill values -> NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].mask(df[num_cols] <= -9999)

    par = df[par_col].where(df[par_col] >= -50)  # remove sensor errors (e.g. -10000)
    rain = df['Prec (mm)'].fillna(0) > 0

    # ---------------- GPP ----------------
    df['GPP_raw'] = df['GPP']
    qc_gpp = np.zeros(len(df), dtype=int)
    gpp = df['GPP']
    qc_gpp |= np.where(gpp.isna(), 1, 0)
    qc_gpp |= np.where(df['flag_NEE_GF'] != 0, 2, 0)          # keep measured NEE only
    qc_gpp |= np.where(df['qc_co2'] > 1, 4, 0)                # Mauder & Foken flag 2
    qc_gpp |= np.where(~(par >= par_min_gpp), 8, 0)           # night / low light (NaN PAR -> flagged)
    qc_gpp |= np.where((gpp < 0) | (gpp > gpp_max), 16, 0)
    relunc = df['GPPsd'] / gpp.where(gpp > 0)
    qc_gpp |= np.where(relunc > gpp_relunc_max, 32, 0)
    if remove_rain:
        qc_gpp |= np.where(rain, 64, 0)
    df['QC_GPP'] = qc_gpp
    df.loc[df['QC_GPP'] != 0, 'GPP'] = np.nan

    # ---------------- SIF ----------------
    sif_cols = [c for c in sif_cols if c in df.columns]
    for c in sif_cols:
        df[c + '_raw'] = df[c]
    ref = 'SIF_3FLD' if 'SIF_3FLD' in df.columns else sif_cols[0]
    sif = df[ref]
    qc_sif = np.zeros(len(df), dtype=int)
    qc_sif |= np.where(sif.isna(), 1, 0)
    qc_sif |= np.where(~(par >= par_min_sif), 2, 0)
    qc_sif |= np.where((sif <= 0) | (sif > sif_max), 4, 0)
    if remove_rain:
        qc_sif |= np.where(rain, 8, 0)

    base_ok = qc_sif == 0
    # consistency between 3FLD and SFM (strongly correlated retrievals)
    if {'SIF_3FLD', 'SIF_SFM_lin_a'}.issubset(df.columns):
        ok = base_ok & df['SIF_SFM_lin_a'].notna()
        if ok.sum() > 20:
            x, y = df.loc[ok, 'SIF_3FLD'].values, df.loc[ok, 'SIF_SFM_lin_a'].values
            slope, intercept = np.polyfit(x, y, 1)
            resid = df['SIF_SFM_lin_a'] - (slope * df['SIF_3FLD'] + intercept)
            r_ok = resid[ok]
            z = _robust_z(resid, np.median(r_ok), np.median(np.abs(r_ok - np.median(r_ok))))
            qc_sif |= np.where(ok & (np.abs(z) > z_thresh), 16, 0)

    # SIF/PAR (apparent yield) outliers within a moving ± window of days
    sif_yield = sif / par
    day = df['DOY_UTC'].astype(int)
    ok = base_ok & np.isfinite(sif_yield)
    yld, dd = sif_yield[ok], day[ok]
    z_yield = pd.Series(np.nan, index=df.index)
    for d0 in np.unique(dd):
        in_win = (dd >= d0 - yield_window_days) & (dd <= d0 + yield_window_days)
        ref_vals = yld[in_win]
        if len(ref_vals) < 10:
            continue
        med = np.median(ref_vals)
        mad = np.median(np.abs(ref_vals - med))
        if mad == 0:
            continue
        idx = yld.index[dd == d0]
        z_yield.loc[idx] = _robust_z(yld.loc[idx], med, mad)
    qc_sif |= np.where(np.abs(z_yield) > z_thresh, 32, 0)

    df['QC_SIF'] = qc_sif
    df['SIF_yield_obs'] = sif_yield.where(df['QC_SIF'] == 0)
    df.loc[df['QC_SIF'] != 0, sif_cols] = np.nan

    if verbose:
        def _summary(name, raw_valid, qc, bits):
            print(f'  {name}: {raw_valid} raw valid -> {(qc == 0).sum()} kept')
            for b, lab in bits.items():
                print(f'      flag {b:>2} {lab:<28}: {((qc & b) > 0)[raw_valid_mask[name]].sum()}')
        raw_valid_mask = {'GPP': df['GPP_raw'].notna(), 'SIF': df[ref + '_raw'].notna()}
        _summary('GPP', raw_valid_mask['GPP'].sum(), df['QC_GPP'].values,
                 {2: 'gap-filled', 4: 'qc_co2>1', 8: 'night/low PAR', 16: 'out of range',
                  32: 'large uncertainty', 64: 'rain'})
        _summary('SIF', raw_valid_mask['SIF'].sum(), df['QC_SIF'].values,
                 {2: 'low PAR', 4: 'out of range', 8: 'rain',
                  16: '3FLD/SFM inconsistent', 32: 'SIF/PAR yield outlier'})
        both = (df['QC_GPP'] == 0) & (df['QC_SIF'] == 0)
        print(f'  GPP & SIF both valid: {both.sum()}')
    return df

# QC flag labels, listed in priority order (used to assign one "main reason" per removed point)
QC_GPP_LABELS = {8: 'night/low PAR', 2: 'gap-filled', 4: 'qc_co2>1', 64: 'rain',
                 16: 'out of range', 32: 'large uncertainty'}
QC_SIF_LABELS = {2: 'low PAR', 8: 'rain', 4: 'out of range',
                 16: '3FLD/SFM inconsistent', 32: 'SIF/PAR yield outlier'}
QC_COLORS = ['#7f7f7f', '#1f77b4', '#9467bd', '#17becf', '#d62728', '#ff7f0e']

def _qc_main_reason(qc, labels):
    """return the label of the first (highest-priority) flag set, 'kept' if qc==0"""
    reason = np.full(len(qc), 'kept', dtype=object)
    todo = qc != 0
    for b, lab in labels.items():
        hit = todo & ((qc & b) > 0)
        reason[hit] = lab
        todo &= ~hit
    reason[todo] = 'other'
    return reason

def _qc_flag_counts(qc, labels):
    """inclusive (flag set) and exclusive (only this flag set) counts"""
    qc = qc & ~1  # ignore 'missing' bit
    incl = [int(((qc & b) > 0).sum()) for b in labels]
    excl = [int((qc == b).sum()) for b in labels]
    return incl, excl

def plot_qc_sif_gpp(df, year, figpath, sif_col='SIF_3FLD', par_col='PAR (µmol/m2/s)', show=False):
    """
    Visualise the output of qc_obs_sif_gpp():
      Fig 1  <year>_QC_kept_vs_removed.jpg : time series and scatter plots, kept vs removed
                                             (removed points coloured by main QC reason)
      Fig 2  <year>_QC_statistics.jpg      : flag counts, monthly / diurnal retention, distributions
    """
    os.makedirs(figpath, exist_ok=True)
    doy = df['DOY_UTC'].values
    hod = df['hour_UTC'].values + df['minute_UTC'].values / 60
    par = df[par_col].where(df[par_col] >= -50).values
    gpp_raw, gpp = df['GPP_raw'].values, df['GPP'].values
    sif_raw, sif = df[sif_col + '_raw'].values, df[sif_col].values
    qg, qs = df['QC_GPP'].values.astype(int), df['QC_SIF'].values.astype(int)
    vg, vs = ~np.isnan(gpp_raw), ~np.isnan(sif_raw)          # raw valid
    rg = _qc_main_reason(qg, QC_GPP_LABELS)
    rs = _qc_main_reason(qs, QC_SIF_LABELS)
    cg = dict(zip(QC_GPP_LABELS.values(), QC_COLORS))
    cs = dict(zip(QC_SIF_LABELS.values(), QC_COLORS))
    kw_rm = dict(s=4, alpha=0.5, lw=0)
    kw_kp = dict(s=4, alpha=0.6, lw=0, c='#2ca02c', label='kept')

    # ================= Fig 1: kept vs removed =================
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    # (a) GPP time series
    a = ax[0, 0]
    for lab, col in cg.items():
        m = vg & (rg == lab)
        if m.any():
            a.scatter(doy[m], gpp_raw[m], color=col, label=f'{lab} ({m.sum()})', **kw_rm)
    m = vg & (qg == 0)
    a.scatter(doy[m], gpp_raw[m], **{**kw_kp, 'label': f'kept ({m.sum()})'})
    a.set(xlabel='DOY (UTC)', ylabel='GPP (µmol m$^{-2}$ s$^{-1}$)', title='(a) GPP time series')
    a.legend(markerscale=3, fontsize=8, ncol=2)
    # (b) SIF time series
    a = ax[0, 1]
    for lab, col in cs.items():
        m = vs & (rs == lab)
        if m.any():
            a.scatter(doy[m], sif_raw[m], color=col, label=f'{lab} ({m.sum()})', **kw_rm)
    m = vs & (qs == 0)
    a.scatter(doy[m], sif_raw[m], **{**kw_kp, 'label': f'kept ({m.sum()})'})
    a.set(xlabel='DOY (UTC)', ylabel=f'{sif_col} (mW m$^{{-2}}$ sr$^{{-1}}$ nm$^{{-1}}$)',
          title=f'(b) {sif_col} time series')
    a.legend(markerscale=3, fontsize=8, ncol=2)
    # (c) GPP vs PAR
    a = ax[1, 0]
    m = vg & (qg != 0)
    a.scatter(par[m], gpp_raw[m], color='lightgrey', label='removed', **kw_rm)
    m = vg & (qg == 0)
    a.scatter(par[m], gpp_raw[m], **kw_kp)
    a.set(xlabel='PAR (µmol m$^{-2}$ s$^{-1}$)', ylabel='GPP', title='(c) GPP light response')
    a.legend(markerscale=3)
    # (d) SIF vs PAR
    a = ax[1, 1]
    for lab, col in cs.items():
        m = vs & (rs == lab)
        if m.any():
            a.scatter(par[m], sif_raw[m], color=col, label=lab, **kw_rm)
    m = vs & (qs == 0)
    a.scatter(par[m], sif_raw[m], **kw_kp)
    a.set(xlabel='PAR (µmol m$^{-2}$ s$^{-1}$)', ylabel=sif_col, title=f'(d) {sif_col} vs PAR')
    a.legend(markerscale=3, fontsize=8)
    # (e) GPP vs SIF, before / after QC
    a = ax[2, 0]
    m_raw = vg & vs & (gpp_raw > 0) & (par >= 50)        # daytime raw pairs
    m_qc = ~np.isnan(gpp) & ~np.isnan(sif)
    a.scatter(sif_raw[m_raw & ~m_qc], gpp_raw[m_raw & ~m_qc], color='lightgrey', label='removed', **kw_rm)
    a.scatter(sif[m_qc], gpp[m_qc], **kw_kp)
    for m, col, lab in [(m_raw, 'grey', 'raw (daytime)'), (m_qc, 'darkgreen', 'after QC')]:
        if m.sum() > 2:
            k, b0 = np.polyfit(sif_raw[m] if lab.startswith('raw') else sif[m],
                               gpp_raw[m] if lab.startswith('raw') else gpp[m], 1)
            r = np.corrcoef(sif_raw[m] if lab.startswith('raw') else sif[m],
                            gpp_raw[m] if lab.startswith('raw') else gpp[m])[0, 1]
            xx = np.linspace(0, np.nanmax(sif_raw[m_raw]), 10)
            a.plot(xx, k * xx + b0, color=col, lw=2,
                   label=f'{lab}: y={k:.1f}x+{b0:.1f}, R²={r**2:.2f}, n={m.sum()}')
    a.set(xlabel=sif_col, ylabel='GPP', title='(e) GPP vs SIF')
    a.legend(markerscale=3, fontsize=8)
    # (f) 3FLD vs SFM consistency
    a = ax[2, 1]
    if 'SIF_SFM_lin_a_raw' in df.columns:
        sfm_raw = df['SIF_SFM_lin_a_raw'].values
        m = vs & ~np.isnan(sfm_raw)
        a.scatter(sif_raw[m & (qs != 0)], sfm_raw[m & (qs != 0)], color='lightgrey', label='removed (other)', **kw_rm)
        mi = m & ((qs & 16) > 0)
        a.scatter(sif_raw[mi], sfm_raw[mi], s=8, color='#d62728', label=f'3FLD/SFM inconsistent ({mi.sum()})')
        a.scatter(sif_raw[m & (qs == 0)], sfm_raw[m & (qs == 0)], **kw_kp)
        lim = [0, np.nanmax([sif_raw[m].max(), sfm_raw[m].max()])]
        a.plot(lim, lim, 'k--', lw=1)
        a.set(xlabel='SIF_3FLD', ylabel='SIF_SFM_lin_a', title='(f) retrieval consistency')
        a.legend(markerscale=3, fontsize=8)
    fig.suptitle(f'Barbeau {year}: observed GPP and SIF, kept vs removed by QC', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figpath, f'{year}_QC_kept_vs_removed.jpg'), dpi=300)
    if not show:
        plt.close(fig)

    # ================= Fig 2: QC statistics =================
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    # (a, b) flag counts: inclusive vs exclusive
    for a, qc, v, labels, name in [(ax[0, 0], qg, vg, QC_GPP_LABELS, 'GPP'),
                                   (ax[0, 1], qs, vs, QC_SIF_LABELS, 'SIF')]:
        incl, excl = _qc_flag_counts(qc[v], labels)
        y = np.arange(len(labels))
        a.barh(y + 0.2, incl, 0.4, color='steelblue', label='flag set (incl. overlap)')
        a.barh(y - 0.2, excl, 0.4, color='orange', label='only this flag')
        for yi, n in zip(y, incl):
            a.text(n, yi + 0.2, f' {n} ({100 * n / v.sum():.1f}%)', va='center', fontsize=8)
        a.set_yticks(y, list(labels.values()))
        a.invert_yaxis()
        n_kept = int((qc[v] == 0).sum())
        a.set(xlabel='number of half-hours',
              title=f'({"a" if name == "GPP" else "b"}) {name} flags: {n_kept}/{v.sum()} kept '
                    f'({100 * n_kept / v.sum():.1f}%)')
        a.legend(fontsize=8, loc='lower right')
    # (c, d) monthly kept / removed by main reason (GPP: daytime only)
    month = df['mois'].values
    for a, v, r, cmap, labels, name in [(ax[1, 0], vg & ((qg & 8) == 0), rg, cg, QC_GPP_LABELS, 'GPP (daytime)'),
                                        (ax[1, 1], vs, rs, cs, QC_SIF_LABELS, 'SIF')]:
        tab = pd.crosstab(month[v], r[v]).reindex(range(1, 13), fill_value=0)
        cols = ['kept'] + [l for l in labels.values() if l in tab.columns]
        bottom = np.zeros(12)
        for c in cols:
            if c not in tab.columns:
                continue
            a.bar(tab.index, tab[c], bottom=bottom, label=c,
                  color='#2ca02c' if c == 'kept' else cmap[c])
            bottom += tab[c].values
        a.set_xticks(range(1, 13))
        a.set(xlabel='month', ylabel='number of half-hours',
              title=f'({"c" if name.startswith("GPP") else "d"}) monthly kept / removed (main reason): {name}')
        a.legend(fontsize=8)
    # (e) diurnal retention fraction
    a = ax[2, 0]
    hbin = np.floor(hod).astype(int)
    for v, qc, col, name in [(vg & ((qg & 8) == 0), qg, '#2ca02c', 'GPP (daytime)'),
                             (vs, qs, '#d62728', 'SIF')]:
        frac = pd.Series(qc[v] == 0).groupby(hbin[v]).mean()
        a.plot(frac.index, 100 * frac.values, 'o-', color=col, label=name)
    a.set(xlabel='hour (UTC)', ylabel='kept fraction (%)', ylim=(0, 105), xlim=(0, 24),
          title='(e) diurnal retention fraction')
    a.legend()
    # (f) distributions before / after QC
    a = ax[2, 1]
    a2 = a.twiny()
    m = vg & (par >= 50) & (gpp_raw > -10)
    bins = np.linspace(np.nanmin(gpp_raw[m]), np.nanmax(gpp_raw[m]), 40)
    a.hist(gpp_raw[m], bins=bins, color='lightgrey', label='GPP raw (daytime)')
    a.hist(gpp[~np.isnan(gpp)], bins=bins, histtype='step', color='#2ca02c', lw=2, label='GPP after QC')
    bins_s = np.linspace(0, np.nanmax(sif_raw[vs]), 40)
    a2.hist(sif_raw[vs], bins=bins_s, histtype='step', color='grey', ls='--', lw=1.5, label='SIF raw')
    a2.hist(sif[~np.isnan(sif)], bins=bins_s, histtype='step', color='#d62728', lw=2, label='SIF after QC')
    a.set(xlabel='GPP (µmol m$^{-2}$ s$^{-1}$)', ylabel='count')
    a2.set_xlabel(f'{sif_col} (mW m$^{{-2}}$ sr$^{{-1}}$ nm$^{{-1}}$)')
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8)
    a.set_title('(f) distributions before / after QC', pad=30)
    fig.suptitle(f'Barbeau {year}: QC statistics of observed GPP and SIF', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figpath, f'{year}_QC_statistics.jpg'), dpi=300)
    if not show:
        plt.close(fig)

def match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers):
    """批量读取CSV，聚合后一次merge"""
    csvfiles_canopy = glob.glob(os.path.join(path_sif_canopy, '*.csv'))
    
    results = []
    for csvfile in csvfiles_canopy:
        df_j = pd.read_csv(csvfile, index_col=False)
        jj_j = int(csvfile.split(os.sep)[-1].split('_')[-5])
        hh_j = int(csvfile.split(os.sep)[-1].split('_')[-4]) / 100 - 1.0 #  offset 1 hour
        
        wl = df_j['wl'].values
        idx_760 = np.where(wl == 760)[0]
        if len(idx_760) == 0:
            continue
        
        sifc760 = df_j.loc[idx_760[0], 'SIFcanopy']
        sift760 = df_j.loc[idx_760[0], 'SIFtotal']
        
        # 读取layers数据
        path_layers = os.path.join(path_sif_layers, csvfile.split(os.sep)[-1].replace('canopy', 'lys_ba'))
        if not os.path.exists(path_layers):
            continue
        
        df_l = pd.read_csv(path_layers, index_col=False)
        if (df_l['JaLAI'].sum() == 0) or (df_l['PBhLAI'].sum() == 0) or (df_l['PARdiraLAI'].sum() < 0):
            continue
        
        # 一次性构建结果行
        row = {'jj': jj_j, 'hh': hh_j, 'SIFcanopy_760nm': sifc760, 'SIFtotal_760nm': sift760}
        # 第1-12列的和
        # for col_idx in range(1, 18):
        for col_idx in range(1, 26):
            col_name = df_l.columns[col_idx]
            row[col_name] = df_l.iloc[:, col_idx].sum()
        
        # 预计算所有中间值
        row.update({
            'PARdiroLAI': df_l.iloc[0, 12],
            'PARdifoLAI': df_l.iloc[0, 13],
            'PBhLAI': df_l['PBhLAI'].sum(),
            'NPQLAI': df_l.iloc[0, 15],
            'qL': df_l.iloc[0, 17],  # the first row's qL
            'phi_F': df_l.iloc[0, 22],  # the first row's phi_F
            'phi_PSII': df_l.iloc[0, 23],  # the first row's phi_PSII
            'VPDest': df_l.iloc[0, 25],  # the first row's VPDest
            'NPQLAI_mean': pd.to_numeric(df_l.iloc[:, 15], errors='coerce').mean(),
            'phi_F_mean': pd.to_numeric(df_l.iloc[:, 22], errors='coerce').mean(),
            'phi_PSII_mean': pd.to_numeric(df_l.iloc[:, 23], errors='coerce').mean(),

            'SIFPSIILAI_yield': pd.to_numeric(df_l.iloc[:, 16], errors='coerce').mean(),
            'SIFPSIILAI_top': df_l.iloc[0, 14],
            'SIFPSIILAI_top5': np.sum(df_l.iloc[0:5, 14]),
            'SIFPSIILAI_yield_top': df_l.iloc[0, 16],
            'SIFPSIILAI_yield_top5': np.mean(pd.to_numeric(df_l.iloc[0:5, 16], errors='coerce')),
            'APARdirect_top': df_l.iloc[0, 10],
            'APARdiffuse_top': df_l.iloc[0, 11],
            'APARtotal_top': df_l.iloc[0, 10] + df_l.iloc[0, 11],
            'APARdiffuse_fraction_top': df_l.iloc[0, 11] / (df_l.iloc[0, 10] + df_l.iloc[0, 11]) if (df_l.iloc[0, 10] + df_l.iloc[0, 11]) > 0 else np.nan,
            'PAR_total': df_l.iloc[0, 12] + df_l.iloc[0, 13],
            'PAR_diffuse_fraction': df_l.iloc[0, 13] / (df_l.iloc[0, 12] + df_l.iloc[0, 13]) if (df_l.iloc[0, 12] + df_l.iloc[0, 13]).sum() > 0 else np.nan
        })
        results.append(row)
    
    # 一次性转df
    df_cas = pd.DataFrame(results)
    
    # 一次merge（替代9000+次逐行赋值）
    if len(df_cas) > 0:
        data_mea = data_mea.merge(df_cas, on=['jj', 'hh'], how='left')
    
    return data_mea

def match_castanea_fast_year(data_mea, year, path_sif_canopy, path_sif_layers):
    """批量读取CSV，聚合后一次merge"""
    csvfiles_canopy = glob.glob(os.path.join(path_sif_canopy, '*.csv'))
    csvfiles_year = [f for f in csvfiles_canopy if f'_SIF_{year}_' in f]
    
    results = []
    for csvfile in csvfiles_year:
        df_j = pd.read_csv(csvfile, index_col=False)
        jj_j = int(csvfile.split(os.sep)[-1].split('_')[-5])
        # hh_j = int(csvfile.split(os.sep)[-1].split('_')[-4]) / 100 - 1.0 #  offset 1 hour (used for decalage = 0)
        hh_j = (int(csvfile.split(os.sep)[-1].split('_')[-4])+30) / 100 - 0.5 #  (used for decalage = -0.8)
        
        wl = df_j['wl'].values
        idx_760 = np.where(wl == 760)[0]
        if len(idx_760) == 0:
            continue
        
        sifc760 = df_j.loc[idx_760[0], 'SIFcanopy']
        sift760 = df_j.loc[idx_760[0], 'SIFtotal']
        
        # 读取layers数据
        path_layers = os.path.join(path_sif_layers, csvfile.split(os.sep)[-1].replace('canopy', 'lys_ba'))
        if not os.path.exists(path_layers):
            continue
        
        df_l = pd.read_csv(path_layers, index_col=False)
        # if (df_l['JaLAI'].sum() == 0) or (df_l['PBhLAI'].sum() == 0) or (df_l['PARdiraLAI'].sum() < 0):
        #     continue
        if (df_l['PBhLAI'].sum() == 0) or (df_l['PARdiraLAI'].sum() < 0):
            continue
        # 一次性构建结果行
        row = {'jj': jj_j, 'hh': hh_j, 'SIFcanopy_760nm': sifc760, 'SIFtotal_760nm': sift760}
        # 第1-12列的和
        # for col_idx in range(1, 18):
        # for col_idx in range(1, 26):
        for col_idx in range(1, 27):
            col_name = df_l.columns[col_idx]
            row[col_name] = df_l.iloc[:, col_idx].sum()
        
        # 预计算所有中间值
        row.update({
            'PARdiroLAI': df_l.iloc[0, 12], # total incident direct PAR of top layer
            'PARdifoLAI': df_l.iloc[0, 13], # total incident diffuse PAR of top layer
            'PARinLAI': df_l.iloc[0, 14], #  input PARin
            'PBhLAI': df_l['PBhLAI'].sum(),
            'NPQLAI': df_l.iloc[0, 16],
            'qL': df_l.iloc[0, 18],  # the first row's qL
            'phi_F': df_l.iloc[0, 23],  # the first row's phi_F
            'phi_PSII': df_l.iloc[0, 24],  # the first row's phi_PSII
            'VPDest': df_l.iloc[0, 26],  # the first row's VPDest
            'NPQLAI_mean': pd.to_numeric(df_l.iloc[:, 16], errors='coerce').mean(),
            'phi_F_mean': pd.to_numeric(df_l.iloc[:, 23], errors='coerce').mean(),
            'phi_PSII_mean': pd.to_numeric(df_l.iloc[:, 24], errors='coerce').mean(),

            'SIFPSIILAI_yield': pd.to_numeric(df_l.iloc[:, 16], errors='coerce').mean(),
            'SIFPSIILAI_top': df_l.iloc[0, 14],
            'SIFPSIILAI_top5': np.sum(df_l.iloc[0:5, 14]),
            'SIFPSIILAI_yield_top': df_l.iloc[0, 16],
            'SIFPSIILAI_yield_top5': np.mean(pd.to_numeric(df_l.iloc[0:5, 16], errors='coerce')),
            'APARdirect_top': df_l.iloc[0, 10],
            'APARdiffuse_top': df_l.iloc[0, 11],
            'APARtotal_top': df_l.iloc[0, 10] + df_l.iloc[0, 11],
            'APARdiffuse_fraction_top': df_l.iloc[0, 11] / (df_l.iloc[0, 10] + df_l.iloc[0, 11]) if (df_l.iloc[0, 10] + df_l.iloc[0, 11]) > 0 else np.nan,
            'PAR_total': df_l.iloc[0, 12] + df_l.iloc[0, 13],
            'PAR_diffuse_fraction': df_l.iloc[0, 13] / (df_l.iloc[0, 12] + df_l.iloc[0, 13]) if (df_l.iloc[0, 12] + df_l.iloc[0, 13]).sum() > 0 else np.nan
        })
        results.append(row)
    
    # 一次性转df
    df_cas = pd.DataFrame(results)
    
    # 一次merge（替代9000+次逐行赋值）
    if len(df_cas) > 0:
        data_mea = data_mea.merge(df_cas, on=['jj', 'hh'], how='left')
    
    return data_mea

# flux data
path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel\data_gpp_with_uncertainties'
path_SIF3 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
path_LIF = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED'
# savepath = r'E:\Datahub\Barbeau\Data_matched_new4_gcc'
savepath = r'E:\Datahub\Barbeau\Data_matched_new7'
if not os.path.exists(savepath):
    os.makedirs(savepath)

# %% match SIF, VIs, LIF with flux data based on DOY
# yearstrs =['2024','2025'] # '2022','2023','2024','2025'
# for yearstr in yearstrs:
#     # read data
#     df_flux = pd.read_excel(glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0], sheet_name='data')
#     df_sif = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_SIFresults_{yearstr}_Yearly_clean.csv'))
#     df_vis = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_VIsresults_{yearstr}_Yearly.csv'))
#     # DOY calculation for flux data (updates all time-related columns in df_flux, including 'an', 'mois', 'jour', 'hh', 'minute', and 'DOY')
#     df_flux = fluxdata_doy_calculation(df_flux)

#     # DOY calculation for SIF data and VIs data
#     df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[1])
#     df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[2])
#     sif_columns = ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
#     # threshold_sif = [6, 3, 4] # mW/m2/sr/nm, set SIF values > threshold to NaN, as they are likely outliers
#     for i, col in enumerate(sif_columns):
#         df_sif.loc[(df_sif[col] > 2) | (df_sif[col] < 0), col] = np.nan # set SIF values > threshold to NaN, as they are likely outliers
#     df_vis['DOY_vis'] = df_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[1])
#     df_vis['Hour'] = df_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[2])

#     # DOY calculation for LIF data
#     # the LIF data already has DOY columns
#     if (yearstr == '2025') or (yearstr == '2024'):
#         df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly_nopulse', f'{yearstr}_LIF.csv'))
#     else:
#         df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))
#     df_lif['DOY_lif'] = df_lif['DOY']
#     df_lif = df_lif.dropna(subset=['DOY_lif']) 
#     df_lif['Hour'] = (df_lif['DOY_lif'] - df_lif['DOY_lif'].astype(int)) * 24

#     # %% match df_match with df_flux based on Time_start
#     dt_start = int(df_flux.loc[0, 'DOY_UTC'])
#     dt_end = int(df_flux.loc[len(df_flux)-1, 'DOY_UTC'])

#     num_max_within_halfhour = 15 # average number of SIF values within a half-hour window
#     num_max_within_halfhour_lif = 1500 # average number of LIF values within a half-hour window

#     # df_flux = match_data_vectorized(yearstr, df_flux, df_sif, df_vis, df_lif, 
#     #                             num_max_within_halfhour, num_max_within_halfhour_lif) # too slow
#     df_flux = match_data_slot(df_flux, df_sif, df_vis, df_lif)
#     df_flux.to_excel(os.path.join(savepath, f'Barbeau_{yearstr}_matched_clean.xlsx'), index=False)


# %% match measured data with modeled data based on DOY
# path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new4'
# path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new5'
path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new7'
savepath = r'E:\Datahub\Barbeau\Data_matched_new7'
# path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_qL\2026.08.27_5pm'
# savepath = r'E:\Datahub\Barbeau\Data_matched_new4_gcc'
path_apar = r'E:\Datahub\Barbeau\Data_flux\Daniel'
data_apar = pd.read_excel(os.path.join(path_apar, 'Barbeau_APAR_20220101-0030_to_20251231-2330.xlsx'))
years = ['2022', '2023','2024', '2025'] # '2022', '2025', '2023','2024',

# path_root = os.path.join(path_sim, 'Res_67_761_LIF_decalage')
# path_root = os.path.join(path_sim, 'Res_67_761_LIF_nodecalage')
path_root = os.path.join(path_sim, 'Res_67_761_LAIlt_full_de_Cabdiff_newNPQparams')

path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
path_sif_layers = os.path.join(path_root, 'SIF_layers')
for year in years:
    data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched_clean.xlsx'))
    print(f'QC of observed SIF and GPP, {year}:')
    data_mea = qc_obs_sif_gpp(data_mea)
    plot_qc_sif_gpp(data_mea, year, os.path.join(savepath, 'figs', 'QC'))

    # path_root = os.path.join(path_sim, f'Res_67_761_{year}_full')
    # path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    # path_sif_layers = os.path.join(path_root, 'SIF_layers')
    # data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
    
    data_mea = match_castanea_fast_year(data_mea, year, path_sif_canopy, path_sif_layers)

    # data_mea.loc[data_mea['SIFPSIILAI_yield_top'] < 0, 'SIFPSIILAI_yield_top'] = np.nan
    # link measured APAR and frac_dif
    data_apar_year = data_apar[data_apar['an'] == int(year)]
    data_mea = data_mea.merge(data_apar_year[['mois', 'jour', 'hour_UTC','minute_UTC', 'APAR_1_35', 'frac_dif']], 
                              on=['mois', 'jour', 'hour_UTC','minute_UTC'], how='left')
    data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_full_APAR.xlsx'), index=False)
    # data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF_APAR.xlsx'), index=False)

    # # data_mea daily mean
    # data_mea['day_UTC'] = data_mea['DOY_UTC'].astype(int)
    # data_mea_daily = (data_mea.groupby('day_UTC', as_index=False).mean(numeric_only=True))
    # data_mea_daily.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_full_dailymean_APAR.xlsx'), index=False)

    # # del data_mea, data_mea_daily
    # data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched_clean.xlsx'))
    # path_root = os.path.join(path_sim, f'Res_67_761_{year}_LIF')
    # path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    # path_sif_layers = os.path.join(path_root, 'SIF_layers')
    # data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
    # # data_mea.loc[data_mea['SIFPSIILAI_yield_top'] < 0, 'SIFPSIILAI_yield_top'] = np.nan
    # data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF.xlsx'), index=False)
    # # link measured APAR and frac_dif
    # data_apar_year = data_apar[data_apar['an'] == int(year)]
    # data_mea = data_mea.merge(data_apar_year[['mois', 'jour', 'hour_UTC','minute_UTC', 'APAR_1_35', 'frac_dif']], on=['mois', 'jour', 'hour_UTC','minute_UTC'], how='left')
    # data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF_APAR.xlsx'), index=False)
    # # # data_mea daily mean
    # # data_mea['day_UTC'] = data_mea['DOY_UTC'].astype(int)
    # # data_mea_daily = data_mea.groupby('day_UTC', as_index=False).mean(numeric_only=True)
    # # data_mea_daily.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF_dailymean_APAR.xlsx'), index=False)
    # # del data_mea, data_mea_daily


# # for 2022 with diffuse fraction
# data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_2022_matched.xlsx'))
# path_root = os.path.join(path_sim, f'Res_67_729_2022_fracdiff')
# path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
# path_sif_layers = os.path.join(path_root, 'SIF_layers')
# data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
# data_mea.to_excel(os.path.join(savepath, f'Barbeau_2022_matched_CASTANEA_fracdiff.xlsx'), index=False)

# # # add measured diffuse fraction to the matched data for validation
# # filenames = ['Barbeau_2022_matched_CASTANEA.xlsx',
# #              'Barbeau_2022_matched_CASTANEA_fracdiff.xlsx',
# #              'Barbeau_2022_matched_CASTANEA_phiF.xlsx']
# filenames = ['Barbeau_2022_matched_CASTANEA.xlsx',
#              'Barbeau_2022_matched_CASTANEA_fracdiff.xlsx']
# data_input_flux = pd.read_excel(r'D:\Projet ifx Castanea\data\meteo\Barbo_2022_with_diffuse_fraction_full.xlsx')
# data_input_flux['hh'] = data_input_flux['minute']/60 + data_input_flux['hour']
# for filename in filenames:
#     data_mea = pd.read_excel(os.path.join(savepath, filename))
#     data_mea['PAR_diffuse_fraction_input'] = np.nan
#     for i in range(len(data_mea)):
#         # idx = (data_input_flux['mois'] == data_mea.loc[i, 'mois']) & \
#         #     (data_input_flux['jour'] == data_mea.loc[i, 'jour']) & \
#         #         (data_input_flux['hh'] == data_mea.loc[i, 'hh'] )
#         idx = (data_input_flux['PAR (µmol/m2/s)'] == data_mea.loc[i, 'PAR (µmol/m2/s)'])
#         if idx.sum() == 1:
#             # print('data_mea: ',data_mea.loc[i, "mois"], data_mea.loc[i, "jour"], data_mea.loc[i, "hh"])
#             # print('data_flux:', data_input_flux.loc[idx, 'mois'].values[0], data_input_flux.loc[idx, 'jour'].values[0], data_input_flux.loc[idx, 'hh'].values[0])
#             # print('PAR values: ', data_input_flux.loc[idx, 'PAR (µmol/m2/s)'].values[0], data_mea.loc[i, 'PAR (µmol/m2/s)'])
#             data_mea.loc[i, 'PAR_diffuse_fraction_input'] = data_input_flux.loc[idx, 'diffuse_fraction'].values[0]
#             data_mea.loc[i, 'PAR_input'] = data_input_flux.loc[idx, 'PAR (µmol/m2/s)'].values[0]
#             data_mea.loc[i, 'fPAR'] = data_input_flux.loc[idx, 'fPAR'].values[0]
#             # print(f'Matched day {data_mea.loc[i, "jour"]} hour {data_mea.loc[i, "hh"]} with diffuse fraction {data_input_flux.loc[idx, "diffuse_fraction"].values[0]:.2f}')

#     # data_mea = data_mea.merge(data_input_flux, on=['PAR (µmol/m2/s)'], how='left')
#     data_mea.to_excel(os.path.join(savepath, filename.replace('.xlsx', '_validation.xlsx')), index=False)

# years = ['2022', '2025'] # '2022', '2025'
# for year in years:
#     data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched.xlsx'))
#     path_root = os.path.join(path_sim, f'Res_67_729_{year}_phiF')
#     path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
#     path_sif_layers = os.path.join(path_root, 'SIF_layers')
    
#     data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
#     data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_phiF.xlsx'), index=False)

# %% validate matched data by plotting time series of GPP, SIF, VIs, and LIF for a few selected days
# def umol_to_mw(umol, wl):
#     # Convert umol/m²/sr/nm to mW/m²/sr/nm
#     # 1 umol/m²/sr/nm = 1e-6 mol/m²/sr/nm
#     # Energy (J) = n (mol) * Avogadro's number * h * c / wl (m)
#     # Power (W) = Energy (J) / time (s)
#     # 1 W = 1 J/s
#     # 1 mW = 1e-3 W
#     # Therefore, mW/m²/sr/nm = umol/m²/sr/nm * 1e-6 * Avogadro's number * h * c / wl (m) * 1e3
#     h = 6.62607015e-34  # Planck constant in J*s
#     c = 2.99792458e8    # Speed of light in m/s
#     avogadro = 6.02214076e23  # Avogadro's number in mol^-1
#     wl_m = wl * 1e-9  # Convert nm to m
#     mw = umol * 1e-6 * avogadro * h * c / wl_m * 1e3
#     return mw

# datapath = r'E:\Datahub\Barbeau\Data_matched_new3'
# savepath = r'E:\Datahub\Barbeau\Data_matched_new3\figs\validation_matching'
# years = ['2025'] # '2022', '2025', '2024'
# for year in years:
#     if not os.path.exists(savepath+os.sep+year):
#         os.makedirs(savepath+os.sep+year)
#     df_matched = pd.read_excel(os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_full.xlsx'))
#     df_matched['SIFcanopy_760nm'] = umol_to_mw(df_matched['SIFcanopy_760nm']/300/np.pi, 760)

#     # df_matched = pd.read_excel(os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_fracdiff.xlsx'))
#     # select a few days for plotting
#     selected_days = range(91,301) # DOY
#     strat = 0.2 
#     for day in selected_days:
#         idx_day = (df_matched['DOY_UTC'] >= day) & (df_matched['DOY_UTC'] < day + 1)
#         fig, ax = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

#         ax[0].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax[0].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR'], marker = 'o',color= 'blue',label='PAR (LIF)')
#         ax[0].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR_total']/strat, marker = 'o',color= 'cyan',label='PAR (input to CASTANEA)', linestyle='dashed')
#         ax[0].set_ylabel('PAR (µmol/m2/s)')
#         ax[0].legend()
#         ax[1].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'GPP'], marker = 'o',color= 'green',label='GPP (flux)')
#         ax[1].set_ylabel('GPP (µmol/m2/s)')
#         ax1 = ax[1].twinx()
#         ax1.plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax1.set_ylabel('PAR (µmol/m2/s)')
#         ax1.legend(loc='upper left')
#         ax[1].legend()
#         ax[2].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'SIF_3FLD'], marker = 'o', color= 'red', label='SIF_3FLD')
#         ax[2].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'SIFcanopy_760nm'], marker = 'o', color= 'b', label='SIF_canopy_760nm', linestyle='dashed')
#         ax[2].set_ylabel('SIF (mW/m2/sr/nm)')
#         ax1 = ax[2].twinx()
#         ax1.plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax1.set_ylabel('PAR (µmol/m2/s)')
#         ax1.legend(loc='upper left')
#         ax[2].legend()
#         ax[3].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'NIRVR'], marker = 'o', color= 'purple', label='NIRvR')
#         ax[3].set_ylabel('NIRvR')
#         ax1 = ax[3].twinx()
#         ax1.plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax1.set_ylabel('PAR (µmol/m2/s)')
#         ax1.legend(loc='upper left')
#         ax[3].legend()
#         ax[4].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'Fs'], marker = 'o', color='k' ,label='Fs')
#         ax[4].set_ylabel('Fs')
#         ax1 = ax[4].twinx()
#         ax1.plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax1.set_ylabel('PAR (µmol/m2/s)')
#         ax1.legend(loc='upper left')
#         ax[4].legend()
#         ax[0].set_title(f'Matched data for DOY (UTC) {day} in {year}', fontsize=16)
#         plt.savefig(os.path.join(savepath, year, f'matched_data_DOY_{day}_{year}.jpg'), dpi=300)
