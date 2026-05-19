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

yearstr ='2022'
# read data
df_flux = pd.read_excel(glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0])
df_sif = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_SIFresults_{yearstr}_Yearly.csv'))
df_vis = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_VIsresults_{yearstr}_Yearly.csv'))
# df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))

# DOY calculation for flux data
df_tmp = df_flux.copy()
df_tmp.loc[df_tmp['hh'] == 0, 'jj'] -= 1
df_tmp.loc[df_tmp['hh'] == 0, 'hh'] = 24
df_tmp['hh'] -= 0.5
df_tmp['DOY'] = df_tmp['jj'] + df_tmp['hh'] / 24
df_flux['DOY'] = df_tmp['DOY']
# DOY calculation for SIF data and VIs data
df_sif['DOY'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
df_sif['Hour'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
sif_columns = ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
threshold_sif = [6, 3, 4] # mW/m2/sr/nm, set SIF values > threshold to NaN, as they are likely outliers
for i, col in enumerate(sif_columns):
    df_sif.loc[(df_sif[col] > threshold_sif[i]) | (df_sif[col] < 0), col] = np.nan # set SIF values > threshold to NaN, as they are likely outliers
    df_sif[col] = df_sif[col] / np.pi # convert to mW/m2/sr/nm

df_vis['DOY'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
df_vis['Hour'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
# DOY calculation for LIF data
# the LIF data already has DOY columns
# df_lif['Hour'] = (df_lif['DOY'] - df_lif['DOY'].astype(int)) * 24

# %% match df_match with df_flux based on Time_start
dt_start = int(df_flux.loc[0, 'DOY'])
dt_end = int(df_flux.loc[len(df_flux)-1, 'DOY'])
hour_start = 7+0.5 
hour_end = 18+0.5
df_flux = df_flux.loc[(df_flux['jj'] >= dt_start) & (df_flux['jj'] <= dt_end) \
                       & (df_flux['hh'] >= hour_start) & (df_flux['hh'] <= hour_end), :]# [7, 18]
df_flux = df_flux.reset_index(drop=True)

for i in range(len(df_flux)):
    day_flux = df_flux.loc[i, 'jj']
    hour_flux = df_flux.loc[i, 'hh']
    # if hour_flux == 0:
    #     day_flux -= 1
    #     hour_flux = 24

    # %% link df_flux and df_sif
    idx_window_sif = (df_sif['DOY'] >= day_flux) & (df_sif['DOY'] < day_flux + 1) \
        & (df_sif['Hour'] >= hour_flux - 0.5) & (df_sif['Hour'] < hour_flux)
    # 3sigma filtering for SIF values in the half-hour window
    df_sif.loc[idx_window_sif, 'SIF_3FLD'] = filter_3sigma(df_sif.loc[idx_window_sif, 'SIF_3FLD'])
    df_sif.loc[idx_window_sif, 'SIF_iFLD'] = filter_3sigma(df_sif.loc[idx_window_sif, 'SIF_iFLD'])
    df_sif.loc[idx_window_sif, 'SIF_SFM_lin_a'] = filter_3sigma(df_sif.loc[idx_window_sif, 'SIF_SFM_lin_a'])
    # don't use mean if more than 80% of values are NaN, as it may not be representative

    num_max_within_halfhour = 10
    nan_ratio = np.sum(df_sif.loc[idx_window_sif, 'SIF_3FLD'].isna()) / 
    if nan_ratio > 0.8:
        print(f'Warning: More than 80% of SIF_3FLD_py values are NaN for day {day_flux} hour {hour_flux}')
        df_flux.loc[i, 'SIF_3FLD_py_mean'] = np.nan
    else:
        df_flux.loc[i, 'SIF_3FLD_py_mean'] = np.nanmean(df_match.loc[idx_window, 'SIF_3FLD_py'])
    
    df_flux.loc[i, ['SIF_3FLD_O2A', 'SIF_iFLD_O2A','SIF_SFM_O2A']] = np.nanmean(df_sif.loc[idx_window_sif, ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']], axis=0)



    print(f'Finished matching SIF for day {day_flux} hour {hour_flux}')


# df_flux['DOY'] = df_flux['jj'] + (df_flux['hh'] - 0.5) / 24.0
# # 将DOY列插入到jj之后
# cols = df_flux.columns.tolist()
# cols.insert(cols.index('jj') + 1, cols.pop(cols.index('DOY')))
# df_flux = df_flux[cols]
# # DOY >=104 and DOY<261
# idx = (df_flux['DOY'] >= 104) & (df_flux['DOY'] < 261)
# df_flux = df_flux.loc[idx, :]
# df_flux = df_flux.reset_index(drop=True)
# # outliers filtering 
# df_flux.loc[(df_flux['GPP'] < 0), 'GPP'] = np.nan
# df_flux.to_csv(savepath + '\Barbeau_2022_LI7500_noAoA_uthvar_hh_DoubleInstrumentGF_with_SIF.csv', index=False)
