# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import datetime
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def sigma_clean_mean(x):
    x = x.to_numpy(dtype=float)
    if x.size == 0:
        return np.nan

    mean = np.nanmean(x)
    std = np.nanstd(x)

    keep = (x >= mean - 3 * std) & (x <= mean + 3 * std)
    x = x.copy()
    x[~keep] = np.nan

    return x

datapath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
years = ['2022','2023','2024','2025']


for iyear, year in enumerate(years):
    print(f'Processing year: {year}')
    yearpath = os.path.join(datapath, year,'PROCESSED\\L2\\Yearly')
    df_sif = pd.read_csv(os.path.join(yearpath, 'PROSIF_SIFresults_' + year + '_Yearly.csv'), index_col=False)

    # time_spend
    df_sif['time_spend'] = (df_sif['Time_end'] - df_sif['Time_start']) * 24 * 60 * 60 # seconds
    idx_time = (df_sif['time_spend'] > 1) & (df_sif['time_spend'] <5.5)
    idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
    df_sif = df_sif[idx_time & idx_value]
    df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
    df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])

    doy_start = int(df_sif['DOY_sif'].min())
    doy_end = int(df_sif['DOY_sif'].max())

    for doy in range(doy_start, doy_end + 1):
        idx_doy = (df_sif['DOY_sif'] >= doy) & (df_sif['DOY_sif'] < doy + 1)
        df_sif.loc[idx_doy, 'SIF_3FLD'] = sigma_clean_mean(df_sif.loc[idx_doy, 'SIF_3FLD'])
        df_sif.loc[idx_doy, 'SIF_iFLD'] = sigma_clean_mean(df_sif.loc[idx_doy, 'SIF_iFLD'])
        df_sif.loc[idx_doy, 'SIF_SFM_lin_a'] = sigma_clean_mean(df_sif.loc[idx_doy, 'SIF_SFM_lin_a'])
    # remove DOY_sif and Hour columns before saving
    df_sif.drop(columns=['time_spend','DOY_sif', 'Hour'], inplace=True)
    df_sif.to_csv(os.path.join(yearpath, 'PROSIF_SIFresults_' + year + '_Yearly_clean.csv'), index=False)

    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(df_sif['DOY_sif'], df_sif['SIF_3FLD'], 'o', markersize=2, label='SIF_3FLD')
    # ax.plot(df_sif['DOY_sif'], df_sif['SIF_3FLD_clean'], 's', markersize=2, label='SIF_3FLD_clean')
    # plt.show()