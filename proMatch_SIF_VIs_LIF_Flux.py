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

yearstrs =['2022','2025']
for yearstr in yearstrs:
    # read data
    df_flux = pd.read_excel(glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0], sheet_name='data')
    df_sif = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_SIFresults_{yearstr}_Yearly.csv'))
    df_vis = pd.read_csv(os.path.join(path_SIF3, yearstr, 'PROCESSED', 'L2', 'Yearly', f'PROSIF_VIsresults_{yearstr}_Yearly.csv'))
    df_lif = pd.read_csv(os.path.join(path_LIF, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv'))

    # DOY calculation for flux data
    df_tmp = df_flux.copy()
    df_tmp.loc[df_tmp['hh'] == 0, 'jj'] -= 1
    df_tmp.loc[df_tmp['hh'] == 0, 'hh'] = 24
    df_tmp['hh'] -= 0.5
    df_tmp['DOY'] = df_tmp['jj'] + df_tmp['hh'] / 24
    df_flux['DOY'] = df_tmp['DOY']
    # DOY calculation for SIF data and VIs data
    df_sif['DOY_sif'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
    df_sif['Hour'] = df_sif['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
    sif_columns = ['SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
    threshold_sif = [6, 3, 4] # mW/m2/sr/nm, set SIF values > threshold to NaN, as they are likely outliers
    for i, col in enumerate(sif_columns):
        df_sif.loc[(df_sif[col] > threshold_sif[i]) | (df_sif[col] < 0), col] = np.nan # set SIF values > threshold to NaN, as they are likely outliers
        df_sif[col] = df_sif[col] / np.pi # convert to mW/m2/sr/nm
    df_vis['DOY_vis'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
    df_vis['Hour'] = df_vis['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
    # DOY calculation for LIF data
    # the LIF data already has DOY columns
    df_lif['DOY_lif'] = df_lif['DOY']
    df_lif['Hour'] = (df_lif['DOY_lif'] - df_lif['DOY_lif'].astype(int)) * 24

    # %% match df_match with df_flux based on Time_start
    dt_start = int(df_flux.loc[0, 'DOY'])
    dt_end = int(df_flux.loc[len(df_flux)-1, 'DOY'])

    # hour_start = 7+0.5 
    # hour_end = 18+0.5
    # df_flux = df_flux.loc[(df_flux['jj'] >= dt_start) & (df_flux['jj'] <= dt_end) \
    #                        & (df_flux['hh'] >= hour_start) & (df_flux['hh'] <= hour_end), :]# [7, 18]
    # df_flux = df_flux.reset_index(drop=True)

    num_max_within_halfhour = 15 # average number of SIF values within a half-hour window
    num_max_within_halfhour_lif = 1500 # average number of LIF values within a half-hour window
    for i in range(len(df_flux)): # len(df_flux) , 9000, 9500
        day_flux = df_flux.loc[i, 'jj']
        hour_flux = df_flux.loc[i, 'hh']
        if hour_flux == 0:
            day_flux -= 1
            hour_flux = 24

        # %% link df_flux and df_sif
        idx_window_sif = (df_sif['DOY_sif'] >= day_flux) & (df_sif['DOY_sif'] < day_flux + 1) \
            & (df_sif['Hour'] >= hour_flux - 0.5) & (df_sif['Hour'] < hour_flux)
        # 3sigma filtering for SIF values in the half-hour window
        sif_columns = ['DOY_sif','SIF_3FLD', 'SIF_iFLD','SIF_SFM_lin_a']
        if np.sum(idx_window_sif) == 0:
            for col in sif_columns:
                df_flux.loc[i, col] = np.nan
        else:
            for col in sif_columns:
                df_sif.loc[idx_window_sif, col] = filter_3sigma(df_sif.loc[idx_window_sif, col])
                # don't use mean if more than 70% of values are NaN, as it may not be representative
                nan_ratio = np.sum(df_sif.loc[idx_window_sif, col].isna()) / num_max_within_halfhour
                if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
                    print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
                    df_flux.loc[i, col] = np.nan
                else:
                    df_flux.loc[i, col] = np.nanmean(df_sif.loc[idx_window_sif, col], axis=0)

        # %% link df_flux and df_vis
        idx_window_vis = (df_vis['DOY_vis'] >= day_flux) & (df_vis['DOY_vis'] < day_flux + 1) \
            & (df_vis['Hour'] >= hour_flux - 0.5) & (df_vis['Hour'] < hour_flux)
        # don't use mean if more than 70% of values are NaN, as it may not be representative
        vis_columns = ['DOY_vis', 'NDVI', 'EVI', 'MTCI', 'CIgreen', 'PRI', 'FCVI', 'NIRv', 'NIRVR']
        if np.sum(idx_window_vis) == 0:
            for col in vis_columns:
                df_flux.loc[i, col] = np.nan
        else:
            for col in vis_columns:
                # 3sigma filtering for VIs values in the half-hour window
                df_vis.loc[idx_window_vis, col] = filter_3sigma(df_vis.loc[idx_window_vis, col])
                nan_ratio = np.sum(df_vis.loc[idx_window_vis, col].isna()) / num_max_within_halfhour
                if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
                    print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
                    df_flux.loc[i, col] = np.nan
                else:
                    df_flux.loc[i, col] = np.nanmean(df_vis.loc[idx_window_vis, col], axis=0)

        # %% link df_flux and df_lif
        idx_window_lif = (df_lif['DOY_lif'] >= day_flux) & (df_lif['DOY_lif'] < day_flux + 1) \
            & (df_lif['Hour'] >= hour_flux - 0.5) & (df_lif['Hour'] < hour_flux)
        # don't use mean if more than 70% of values are NaN, as it may not be representative
        lif_columns = ['DOY_lif', 'Fs', 'PAR', 'Fcont']
        if np.sum(idx_window_lif) == 0:
            for col in lif_columns:
                df_flux.loc[i, col] = np.nan
        else:
            for col in lif_columns:
                # 3sigma filtering for LIF values in the half-hour window
                df_lif.loc[idx_window_lif, col] = filter_3sigma(df_lif.loc[idx_window_lif, col])
                nan_ratio = np.sum(df_lif.loc[idx_window_lif, col].isna()) / num_max_within_halfhour_lif
                if nan_ratio > 0.7: # less than 30% of values are valid, don't use mean
                    print(f'Warning: More than 70% of {col} values are NaN for day {day_flux} hour {hour_flux}')
                    df_flux.loc[i, col] = np.nan
                else:
                    df_flux.loc[i, col] = np.nanmean(df_lif.loc[idx_window_lif, col], axis=0)
        
        print(f'Finished matching SIF for day {day_flux} hour {hour_flux}')

    df_flux.to_excel(os.path.join(savepath, f'Barbeau_{yearstr}_matched.xlsx'), index=False)