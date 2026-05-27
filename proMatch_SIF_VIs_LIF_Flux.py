# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def matlab_datenum_to_datetime(yearstr, datenum, offset_hours=5):
    # MATLAB datenum 是从 0000-01-01 开始的天数，Python datetime 从 1970-01-01 开始
    # MATLAB datenum 的整数部分是天数，小数部分是一天中的时间
    # days = int(datenum)
    # fraction = datenum - days
    # dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    if yearstr == '2025':
        offset_hours = 2 # 2025年数据的时间已经是UTC时间，不需要再加上5小时的偏移
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

def match_data_slot(df_flux, df_sif, df_vis, df_lif):
    # 1) flux 也做 slot
    df_flux = add_halfhour_slot(df_flux, "DOY_UTC")

    # 2) 三类数据先按 slot 聚合
    df_sif["slot"] = np.floor(df_sif["DOY_sif"] * 48).astype("int64")
    df_vis["slot"] = np.floor(df_vis["DOY_vis"] * 48).astype("int64")
    df_lif["slot"] = np.floor(df_lif["DOY_lif"] * 48).astype("int64")

    sif_cols = ["DOY_sif","SIF_3FLD", "SIF_iFLD", "SIF_SFM_lin_a"]
    vis_cols = ["DOY_vis","NDVI", "EVI", "MTCI", "CIgreen", "PRI", "FCVI", "NIRv", "mNDI705", "NIRVR"]
    lif_cols = ["DOY_lif","Fs", "PAR", "Fcont"]

    df_sif_agg = df_sif.groupby("slot", as_index=False)[sif_cols].agg(
        lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
    )
    df_vis_agg = df_vis.groupby("slot", as_index=False)[vis_cols].agg(
        lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
    )
    df_lif_agg = df_lif.groupby("slot", as_index=False)[lif_cols].agg(
        lambda s: sigma_clean_mean_with_ratio(s, min_valid_ratio=0.3)
    )

    # 3) 一次 merge 回 flux
    df_flux = df_flux.merge(df_sif_agg, on="slot", how="left")
    df_flux = df_flux.merge(df_vis_agg, on="slot", how="left")
    df_flux = df_flux.merge(df_lif_agg, on="slot", how="left")
    return df_flux


# flux data
path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel\data_gpp_with_uncertainties'
path_SIF3 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
path_LIF = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED'
savepath = r'E:\Datahub\Barbeau\Data_matched'
if not os.path.exists(savepath):
    os.makedirs(savepath)

# %% match SIF, VIs, LIF with flux data based on DOY
# yearstrs =['2022'] # '2022','2025'
# for yearstr in yearstrs:
#     # read data
#     df_flux = pd.read_excel(glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0], sheet_name='data')
#     df_sif = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_SIFresults_{yearstr}_Yearly.csv'))
#     df_vis = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_VIsresults_{yearstr}_Yearly.csv'))
#     df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))
#     # DOY calculation for flux data (updates all time-related columns in df_flux, including 'an', 'mois', 'jour', 'hh', 'minute', and 'DOY')
#     df_flux = fluxdata_doy_calculation(df_flux)

#     # DOY calculation for SIF data and VIs data
#     df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[1])
#     df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[2])
#     sif_columns = ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
#     threshold_sif = [6, 3, 4] # mW/m2/sr/nm, set SIF values > threshold to NaN, as they are likely outliers
#     for i, col in enumerate(sif_columns):
#         df_sif.loc[(df_sif[col] > threshold_sif[i]) | (df_sif[col] < 0), col] = np.nan # set SIF values > threshold to NaN, as they are likely outliers
#         df_sif[col] = df_sif[col] / np.pi # convert to mW/m2/sr/nm
#     df_vis['DOY_vis'] = df_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[1])
#     df_vis['Hour'] = df_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[2])

#     # DOY calculation for LIF data
#     # the LIF data already has DOY columns
#     df_lif['DOY_lif'] = df_lif['DOY']
#     df_lif['Hour'] = (df_lif['DOY_lif'] - df_lif['DOY_lif'].astype(int)) * 24

#     # %% match df_match with df_flux based on Time_start
#     dt_start = int(df_flux.loc[0, 'DOY_UTC'])
#     dt_end = int(df_flux.loc[len(df_flux)-1, 'DOY_UTC'])

#     num_max_within_halfhour = 15 # average number of SIF values within a half-hour window
#     num_max_within_halfhour_lif = 1500 # average number of LIF values within a half-hour window

#     # df_flux = match_data_vectorized(yearstr, df_flux, df_sif, df_vis, df_lif, 
#     #                             num_max_within_halfhour, num_max_within_halfhour_lif)
#     df_flux = match_data_slot(df_flux, df_sif, df_vis, df_lif)
#     df_flux.to_excel(os.path.join(savepath, f'Barbeau_{yearstr}_matched.xlsx'), index=False)


# %% match measured data with modeled data based on DOY
path_sim = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new'
savepath = r'E:\Datahub\Barbeau\Data_matched'
years = ['2022'] # '2022', '2025'
for year in years:
    data_mea = pd.read_excel(os.path.join(savepath, f'Barbeau_{year}_matched.xlsx'))
    path_root = os.path.join(path_sim, f'Res_67_729_{year}')
    # %% 提取SIF760nm并与对应时间的GPP数据进行匹配
    # 提取SIF760nm数据
    path_sif_layers = os.path.join(path_root, 'SIF_layers')
    path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    csvfiles_canopy = glob.glob(os.path.join(path_sif_canopy, '*.csv'))

    sifres = []
    for j, csvfile in enumerate(csvfiles_canopy):
        df_j = pd.read_csv(csvfile, index_col=False)
        jj_j = int(csvfile.split(os.sep)[-1].split('_')[-5])
        hh_j = int(csvfile.split(os.sep)[-1].split('_')[-4])/100
        
        wl = df_j['wl']
        idx = np.where(wl==760)
        sifc760 = df_j.loc[idx, 'SIFcanopy'].values[0]
        sift760 = df_j.loc[idx, 'SIFtotal'].values[0]
        # 对应的layers的数据
        path_layers = os.path.join(path_sif_layers, csvfile.split(os.sep)[-1].replace('canopy','lys_ba'))
        df_l = pd.read_csv(path_layers, index_col=False)
        if (sum(df_l['JaLAI']) == 0) or (sum(df_l['PBhLAI']) == 0) or (sum(df_l['PARdiraLAI']) < 0):
            print('Warning: JaLAI or PBhLAI is zero for file: '+csvfile)
            continue
        
        # match with GPP data
        idx_m = (data_mea['jj'] == jj_j) & (data_mea['hh'] == hh_j)
        if len(idx_m) > 0:
            # 第1-9列，每列各取和，保存在tmp_sum中（使用list避免只读数组问题）
            tmp_sum = df_l.iloc[:,1:12].sum().tolist()
            tmp_sum[10] = df_l.iloc[0,11] # qL列取top layer的值
            tmp_sum[9] = df_l.iloc[:,10].mean() # SIFPSIILAI_yield列取均值
            tmp_sum[4] = df_l.iloc[0,5] # PARdirect
            tmp_sum[5] = df_l.iloc[0,6] # PARdiffuse
            # 与LIF相关及顶层PAR的统计值追加到tmp_sum
            tmp_sum.extend([
                df_l.iloc[0,8],  # top layer的SIFPSIILAI值
                np.sum(df_l.iloc[0:5,8]),  # top 5 layers的SIFPSIILAI的和
                df_l.iloc[0,10],  # top layer的SIFPSIILAI_yield值
                np.mean(df_l.iloc[0:5,10]),  # top 5 layers的SIFPSIILAI_yield的均值
                df_l.iloc[0,3],  # top layer的APARdirect值
                df_l.iloc[0,4],  # top layer的APARdiffuse值
                df_l.iloc[0,3] + df_l.iloc[0,4],  # top layer的APARtotal值 = APAR total
                df_l.iloc[0,4] / (df_l.iloc[0,3] + df_l.iloc[0,4]),  # top layer的APARdiffuse fraction值
                tmp_sum[2] + tmp_sum[3],  # APAR total
                df_l.iloc[0,5] + df_l.iloc[0,6], # PAR total
                df_l.iloc[0,5] / (df_l.iloc[0,5] + df_l.iloc[0,6]) # incident PAR diffuse fraction
            ])
            # 定义tmp_columns，包含第1-9列的列名，以及与LIF相关的两列的列名
            tmp_columns = df_l.columns[1:12].tolist() + ['SIFPSIILAI_top', 'SIFPSIILAI_top5', 
                                                        'SIFPSIILAI_yield_top', 'SIFPSIILAI_yield_top5',
                                                        'APARdirect_top', 'APARdiffuse_top', 
                                                        'APARtotal_top', 'APARdiffuse_fraction_top',
                                                        'APARtotal', 'PARtotal', 
                                                        'incident_PAR_diffuse_fraction']
            # data_mea.loc[idx_m, 'DOY_cas'] = doy_f
            # 如果tmp_sum中的值不是数值类型，则转换为数值类型（如果无法转换，则设置为NaN）
            tmp_sum = [pd.to_numeric(x, errors='coerce') for x in tmp_sum]
            # fill the matched data into data_mea
            data_mea.loc[idx_m, 'SIFcanopy_760nm'] = sifc760
            data_mea.loc[idx_m, 'SIFtotal_760nm'] = sift760
            data_mea.loc[idx_m, tmp_columns] = tmp_sum
        else:
            print('Warning: No matching GPP data for file: '+csvfile)
            # data_mea.loc[idx_m, 'DOY_cas'] = np.nan
            data_mea.loc[idx_m, 'SIFcanopy_760nm'] = np.nan
            data_mea.loc[idx_m, 'SIFtotal_760nm'] = np.nan
            data_mea.loc[idx_m, tmp_columns] = [np.nan]*len(tmp_columns)

    data_mea.to_excel(os.path.join(savepath, f'Barbeau_{year}_matched_CASTANEA.xlsx'), index=False)

# %% validate matched data by plotting time series of GPP, SIF, VIs, and LIF for a few selected days
# datapath = r'E:\Datahub\Barbeau\Data_matched'
# savepath = r'E:\Datahub\Barbeau\Data_matched\figs'
# years = ['2022'] # '2022', '2025'
# for year in years:
#     if not os.path.exists(savepath+os.sep+year):
#         os.makedirs(savepath+os.sep+year)
#     df_matched = pd.read_excel(os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA.xlsx'))
#     # select a few days for plotting
#     selected_days = range(91,365) # DOY
#     for day in selected_days:
#         idx_day = (df_matched['DOY_UTC'] >= day) & (df_matched['DOY_UTC'] < day + 1)
#         fig, ax = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

#         ax[0].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], marker = 'o',color= 'orange',label='PAR (flux)')
#         ax[0].plot(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day, 'PAR'], marker = 'o',color= 'blue',label='PAR (LIF)')
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
