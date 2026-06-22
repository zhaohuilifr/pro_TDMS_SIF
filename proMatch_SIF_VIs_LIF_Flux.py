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
    vis_cols = ["DOY_vis","NDVI", "EVI", "MTCI", "CIgreen", "PRI", "FCVI", "NIRv", "mNDI705", "NIRVR"]
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
        for col_idx in range(1, 18):
            col_name = df_l.columns[col_idx]
            row[col_name] = df_l.iloc[:, col_idx].sum()
        
        # 预计算所有中间值
        row.update({
            'PARdiroLAI': df_l.iloc[0, 12],
            'PARdifoLAI': df_l.iloc[0, 13],
            'PBhLAI': df_l['PBhLAI'].sum(),
            'qL': df_l.iloc[0, 17],  # the first row's qL, use PAR of the first row too
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
savepath = r'E:\Datahub\Barbeau\Data_matched_new3'
if not os.path.exists(savepath):
    os.makedirs(savepath)

 # %% match SIF, VIs, LIF with flux data based on DOY
# yearstrs =['2022','2023','2024','2025'] # '2022','2023','2024','2025'
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
#     df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))
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
path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new3'
savepath = r'E:\Datahub\Barbeau\Data_matched_new3'
path_apar = r'E:\Datahub\Barbeau\Data_flux\Daniel'
data_apar = pd.read_excel(os.path.join(path_apar, 'Barbeau_APAR_20220101-0030_to_20251231-2330.xlsx'))
years = ['2022','2023','2024', '2025'] # '2022', '2025'
for year in years:
    data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched_clean.xlsx'))
    path_root = os.path.join(path_sim, f'Res_67_761_{year}_full')
    path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    path_sif_layers = os.path.join(path_root, 'SIF_layers')
    data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
    data_mea.loc[data_mea['SIFPSIILAI_yield_top'] < 0, 'SIFPSIILAI_yield_top'] = np.nan
    # link measured APAR and frac_dif
    data_apar_year = data_apar[data_apar['an'] == int(year)]
    data_mea = data_mea.merge(data_apar_year[['mois', 'jour', 'hour_UTC','minute_UTC', 'APAR_1_35', 'frac_dif']], on=['mois', 'jour', 'hour_UTC','minute_UTC'], how='left')
    data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_full_APAR.xlsx'), index=False)
    # data_mea daily mean
    data_mea['day_UTC'] = data_mea['DOY_UTC'].astype(int)
    data_mea_daily = (data_mea.groupby('day_UTC', as_index=False).mean(numeric_only=True))
    data_mea_daily.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_full_dailymean_APAR.xlsx'), index=False)

    # del data_mea, data_mea_daily
    data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched_clean.xlsx'))
    path_root = os.path.join(path_sim, f'Res_67_761_{year}_LIF')
    path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    path_sif_layers = os.path.join(path_root, 'SIF_layers')
    data_mea = match_castanea_fast(data_mea, path_sif_canopy, path_sif_layers)
    data_mea.loc[data_mea['SIFPSIILAI_yield_top'] < 0, 'SIFPSIILAI_yield_top'] = np.nan
    data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF.xlsx'), index=False)
    # link measured APAR and frac_dif
    data_apar_year = data_apar[data_apar['an'] == int(year)]
    data_mea = data_mea.merge(data_apar_year[['mois', 'jour', 'hour_UTC','minute_UTC', 'APAR_1_35', 'frac_dif']], on=['mois', 'jour', 'hour_UTC','minute_UTC'], how='left')
    data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF_APAR.xlsx'), index=False)
    # data_mea daily mean
    data_mea['day_UTC'] = data_mea['DOY_UTC'].astype(int)
    data_mea_daily = data_mea.groupby('day_UTC', as_index=False).mean(numeric_only=True)
    data_mea_daily.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA_LIF_dailymean_APAR.xlsx'), index=False)
    del data_mea, data_mea_daily



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
