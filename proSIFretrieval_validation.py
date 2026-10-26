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

datapath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
years = ['2022','2023','2024','2025']


fig, ax = plt.subplots(4, 1, figsize=(18, 10))
for iyear, year in enumerate(years):
    print(f'Processing year: {year}')
    yearpath = os.path.join(datapath, year,'PROCESSED\\L2\\Yearly')
    df_sif = pd.read_csv(os.path.join(yearpath, 'PROSIF_SIFresults_' + year + '_Yearly.csv'), index_col=False)

    # time_spend
    df_sif['time_spend'] = (df_sif['Time_end'] - df_sif['Time_start']) * 24 * 60 * 60 # seconds
    # idx_time = (df_sif['time_spend'] > 1) & (df_sif['time_spend'] <5.5)
    # idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
    # df_sif = df_sif[idx_time & idx_value]

    idx_time = (df_sif['time_spend'] >= 0) & (df_sif['time_spend'] <5.5)
    idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
    df_sif = df_sif[idx_value & idx_time]

    df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
    df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])

    # plot doy vs SIF_3FLD
    sc = ax[iyear].scatter(df_sif['DOY_sif'], df_sif['rad760'], s=4, c=df_sif['SIF_3FLD'], cmap='viridis', label='SIF_3FLD')
    fig.colorbar(sc, ax=ax[iyear], label='SIF_3FLD')
    ax[iyear].set_xlabel('Day of Year')
    ax[iyear].set_ylabel('rad760')
    ax[iyear].set_title(f'Year {year}')
    ax[iyear].grid()
    ax[iyear].set_xlim(100, 300)
    ax[iyear].set_ylim(0, 10)
    ax[iyear].legend()

plt.show()