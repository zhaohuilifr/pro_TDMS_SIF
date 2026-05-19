# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def matlab_datenum_to_datetime(datenum):
    # MATLAB datenum 是从 0000-01-01 开始的天数，Python datetime 从 1970-01-01 开始
    # MATLAB datenum 的整数部分是天数，小数部分是一天中的时间
    # days = int(datenum)
    # fraction = datenum - days
    # dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    dt = pd.to_datetime(datenum - 719529, unit='D')
    hour = dt.hour + dt.minute / 60 + dt.second / 3600 + 7.0  # offset 7 hours
    doy = dt.timetuple().tm_yday + hour / 24
    return dt, doy, hour

def filter_3sigma(array):
    mean = np.nanmean(array)
    std = np.nanstd(array)
    array[~(array >= mean - 3 * std) & (array <= mean + 3 * std)] = np.nan
    if np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)) > 0:
        print('number of outliers filtered:', np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)))
    return array

# flux data
path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel'
path_SIF3 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
path_LIF = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED'
savepath = r'E:\Datahub\Barbeau\Data_matched'
if not os.path.exists(savepath):
    os.makedirs(savepath)

# %% match SIF, VIs, LIF with flux data based on DOY
# yearstrs =['2022','2025']
# for yearstr in yearstrs:
#     # read data
#     df_flux = pd.read_excel(glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0], sheet_name='data')
#     df_sif = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_SIFresults_{yearstr}_Yearly.csv'))
#     df_vis = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_VIsresults_{yearstr}_Yearly.csv'))
#     df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))

#     # DOY calculation for flux data
#     df_tmp = df_flux.copy()
#     df_tmp.loc[df_tmp['hh'] == 0, 'jj'] -= 1
#     df_tmp.loc[df_tmp['hh'] == 0, 'hh'] = 24
#     df_tmp['hh'] -= 0.5
#     df_tmp['DOY'] = df_tmp['jj'] + df_tmp['hh'] / 24
#     df_flux['DOY'] = df_tmp['DOY']
#     # DOY calculation for SIF data and VIs data
#     df_sif['DOY_sif'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
#     df_sif['Hour'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
#     sif_columns = ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
#     threshold_sif = [6, 3, 4] # mW/m2/sr/nm, set SIF values > threshold to NaN, as they are likely outliers
#     for i, col in enumerate(sif_columns):
#         df_sif.loc[(df_sif[col] > threshold_sif[i]) | (df_sif[col] < 0), col] = np.nan # set SIF values > threshold to NaN, as they are likely outliers
#         df_sif[col] = df_sif[col] / np.pi # convert to mW/m2/sr/nm
#     df_vis['DOY_vis'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
#     df_vis['Hour'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
#     # DOY calculation for LIF data
#     # the LIF data already has DOY columns
#     df_lif['DOY_lif'] = df_lif['DOY']
#     df_lif['Hour'] = (df_lif['DOY_lif'] - df_lif['DOY_lif'].astype(int)) * 24

#     # %% match df_match with df_flux based on Time_start
#     dt_start = int(df_flux.loc[0, 'DOY'])
#     dt_end = int(df_flux.loc[len(df_flux)-1, 'DOY'])

#     # hour_start = 7+0.5 
#     # hour_end = 18+0.5
#     # df_flux = df_flux.loc[(df_flux['jj'] >= dt_start) & (df_flux['jj'] <= dt_end) \
#     #                        & (df_flux['hh'] >= hour_start) & (df_flux['hh'] <= hour_end), :]# [7, 18]
#     # df_flux = df_flux.reset_index(drop=True)

#     num_max_within_halfhour = 15 # average number of SIF values within a half-hour window
#     num_max_within_halfhour_lif = 1500 # average number of LIF values within a half-hour window
#     for i in range(len(df_flux)): # len(df_flux) , 9000, 9500
#         day_flux = df_flux.loc[i, 'jj']
#         hour_flux = df_flux.loc[i, 'hh']
#         if hour_flux == 0:
#             day_flux -= 1
#             hour_flux = 24

#         # %% link df_flux and df_sif
#         idx_window_sif = (df_sif['DOY_sif'] >= day_flux) & (df_sif['DOY_sif'] < day_flux + 1) \
#             & (df_sif['Hour'] >= hour_flux - 0.5) & (df_sif['Hour'] < hour_flux)
#         # 3sigma filtering for SIF values in the half-hour window
#         sif_columns = ['DOY_sif','SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
#         if np.sum(idx_window_sif) == 0:
#             for col in sif_columns:
#                 df_flux.loc[i, col] = np.nan
#         else:
#             for col in sif_columns:
#                 df_sif.loc[idx_window_sif, col] = filter_3sigma(df_sif.loc[idx_window_sif, col])
#                 # don't use mean if more than 70% of values are NaN, as it may not be representative
#                 nan_ratio = np.sum(df_sif.loc[idx_window_sif, col].isna()) / num_max_within_halfhour
#                 if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
#                     print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
#                     df_flux.loc[i, col] = np.nan
#                 else:
#                     df_flux.loc[i, col] = np.nanmean(df_sif.loc[idx_window_sif, col], axis=0)

#         # %% link df_flux and df_vis
#         idx_window_vis = (df_vis['DOY_vis'] >= day_flux) & (df_vis['DOY_vis'] < day_flux + 1) \
#             & (df_vis['Hour'] >= hour_flux - 0.5) & (df_vis['Hour'] < hour_flux)
#         # don't use mean if more than 70% of values are NaN, as it may not be representative
#         vis_columns = ['DOY_vis', 'NDVI', 'EVI', 'MTCI', 'CIgreen', 'PRI', 'FCVI', 'NIRv', 'NIRVR']
#         if np.sum(idx_window_vis) == 0:
#             for col in vis_columns:
#                 df_flux.loc[i, col] = np.nan
#         else:
#             for col in vis_columns:
#                 # 3sigma filtering for VIs values in the half-hour window
#                 df_vis.loc[idx_window_vis, col] = filter_3sigma(df_vis.loc[idx_window_vis, col])
#                 nan_ratio = np.sum(df_vis.loc[idx_window_vis, col].isna()) / num_max_within_halfhour
#                 if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
#                     print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
#                     df_flux.loc[i, col] = np.nan
#                 else:
#                     df_flux.loc[i, col] = np.nanmean(df_vis.loc[idx_window_vis, col], axis=0)

#         # %% link df_flux and df_lif
#         idx_window_lif = (df_lif['DOY_lif'] >= day_flux) & (df_lif['DOY_lif'] < day_flux + 1) \
#             & (df_lif['Hour'] >= hour_flux - 0.5) & (df_lif['Hour'] < hour_flux)
#         # don't use mean if more than 70% of values are NaN, as it may not be representative
#         lif_columns = ['DOY_lif', 'Fs', 'PAR', 'Fcont']
#         if np.sum(idx_window_lif) == 0:
#             for col in lif_columns:
#                 df_flux.loc[i, col] = np.nan
#         else:
#             for col in lif_columns:
#                 # 3sigma filtering for LIF values in the half-hour window
#                 df_lif.loc[idx_window_lif, col] = filter_3sigma(df_lif.loc[idx_window_lif, col])
#                 nan_ratio = np.sum(df_lif.loc[idx_window_lif, col].isna()) / num_max_within_halfhour_lif
#                 if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
#                     print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
#                     df_flux.loc[i, col] = np.nan
#                 else:
#                     df_flux.loc[i, col] = np.nanmean(df_lif.loc[idx_window_lif, col], axis=0)
        
#         print(f'Finished matching SIF for day {day_flux} hour {hour_flux}')

#     df_flux.to_excel(os.path.join(savepath, f'Barbeau_{yearstr}_matched.xlsx'), index=False)

# %% match measured data with modeled data based on DOY
data_mea = pd.read_excel(os.path.join(savepath, 'Barbeau_2022_matched.xlsx'))

folders = ['Res_67_729_new'] #['Res_67_300', 'Res_67_400', 'Res_67_500', 'Res_67_600'] #Res_67_1000_AEF_m2
# folders = ['Res_67_729_150-250', 'Res_67_729_AEF_m1','Res_67_729_AEF_m2',
#            'Res_67_729_Allband_AEF_m1','Res_67_729_Allband_AEF_m2'] #'Res_67_729_150-250',
# folders = os.listdir(r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_ql_sensitivity_LIF455nm\Rawdata')
for folder in folders:
    path_root = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis' + os.sep + folder
    # path_root = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_ql_sensitivity_LIF455nm\Rawdata' + os.sep + folder
    path_sif_layers = os.path.join(path_root, 'SIF_layers')
    path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    path_cas_output = os.path.join(path_root, 'Cas_Outputs')
    path_save_figs = os.path.join(path_root, 'Figs')
    path_save_file = os.path.join(path_root, 'Analysis')
    if not os.path.exists(path_save_figs):
        os.makedirs(path_save_figs)
    if not os.path.exists(path_save_file):
        os.makedirs(path_save_file)
    print(f'Paths set for {folder}.')

    # %% 提取SIF760nm并与对应时间的GPP数据进行匹配
    # 提取SIF760nm数据
    path_sif_layers = os.path.join(path_root, 'SIF_layers')
    path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
    csvfiles_canopy = glob.glob(os.path.join(path_sif_canopy, '*.csv'))

    sifres = []
    for j, csvfile in enumerate(csvfiles_canopy):
        df_j = pd.read_csv(csvfile, index_col=False)
        doy_j = csvfile.split(os.sep)[-1].split('_')[-5]
        mint_j = csvfile.split(os.sep)[-1].split('_')[-4][-2:]
        if mint_j == '04':
            mint_j = '00'
        elif mint_j == '54':
            mint_j = '30'
        hour_j = csvfile.split(os.sep)[-1].split('_')[-4][:-2]
        # doy_f = int(doy_j) + int(hour_j)/24 + int(mint_j)/(24*60) 
        # # CASTANEA输出的时间是半小时0点，FluXNET的时间是半小时整点，所以需要减去0.5小时？ 
        # 可能是这个原因，后面匹配数据时，需要注意这个时间差，或者直接在CASTANEA输出的文件名中把时间改成半小时整点
        doy_f = int(doy_j) + int(hour_j)/24.0 + int(mint_j)/(24.0*60.0) - (30.0/(24.0*60.0)) 
        
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
        # 第1-9列，每列各取和，保存在tmp_sum中（使用list避免只读数组问题）
        tmp_sum = df_l.iloc[:,1:12].sum().tolist()
        # 第9列取均值，替换掉tmp_sum中的对应值
        # tmp_sum[8] = df_l.iloc[:,9].mean() # qL列取均值
        # tmp_sum[4] = df_l.iloc[0,5] # PARdiraLAI列取top layer的值
        # tmp_sum[5] = df_l.iloc[0,6] # PARdifaLAI列取top layer的值
        tmp_sum[10] = df_l.iloc[0,11] # qL列取top layer的值
        tmp_sum[9] = df_l.iloc[:,10].mean() # SIFPSIILAI_yield列取均值
        # 与LIF相关及顶层PAR的统计值追加到tmp_sum
        tmp_sum.extend([
            df_l.iloc[0,8],  # top layer的SIFPSIILAI值
            np.sum(df_l.iloc[0:5,8]),  # top 5 layers的SIFPSIILAI的和
            df_l.iloc[0,10],  # top layer的SIFPSIILAI_yield值
            np.mean(df_l.iloc[0:5,10]),  # top 5 layers的SIFPSIILAI_yield的均值
            df_l.iloc[0,3],  # top layer的APARdirect值
            df_l.iloc[0,4],  # top layer的APARdiffuse值
            df_l.iloc[0,3] + df_l.iloc[0,4],  # top layer的APARtotal值
            df_l.iloc[0,4] / (df_l.iloc[0,3] + df_l.iloc[0,4]),  # top layer的APARdiffuse fraction值
        ])
        # 定义tmp_columns，包含第1-9列的列名，以及与LIF相关的两列的列名
        tmp_columns = df_l.columns[1:12].tolist() + ['SIFPSIILAI_top', 'SIFPSIILAI_top5', 
                                                    'SIFPSIILAI_yield_top', 'SIFPSIILAI_yield_top5']+ \
                                                        ['APARdirect_top', 'APARdiffuse_top', 'APARtotal_top', 'APARdiffuse_fraction_top']
        # match with GPP data
        idx_m = np.where(np.isclose(data_mea['DOY'].values, doy_f, atol=5/60/24))[0] # within 5 minutes
        if len(idx_m) > 0:
            data_mea.loc[idx_m, 'DOY_cas'] = doy_f
            data_mea.loc[idx_m, 'SIFcanopy_760nm'] = sifc760
            data_mea.loc[idx_m, 'SIFtotal_760nm'] = sift760
            data_mea.loc[idx_m, tmp_columns] = tmp_sum
        else:
            print('Warning: No matching GPP data for file: '+csvfile)
            data_mea.loc[idx_m, 'DOY_cas'] = np.nan
            data_mea.loc[idx_m, 'SIFcanopy_760nm'] = np.nan
            data_mea.loc[idx_m, 'SIFtotal_760nm'] = np.nan
            data_mea.loc[idx_m, tmp_columns] = [np.nan]*len(tmp_columns)

    # sifres['phi_F_sim(apparent)'] = sifres['SIFcanopy_760nm'] / sifres['PAR']
    # sifres['phi_F_mea(apparent)'] = sifres['SIF_mea'] / sifres['PAR']
    # sifres.loc[sifres['phi_F_mea(apparent)']>0.004,'phi_F_mea(apparent)'] = np.nan # outlier removal
    # 按DOY_mea排序
    # sifres = sifres.sort_values('DOY_mea').reset_index(drop=True)
    data_mea.to_csv(os.path.join(savepath, f'SIF_GPP_matching_{path_root.split(os.sep)[-1]}.csv'), index=False)