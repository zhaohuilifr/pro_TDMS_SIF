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


# %% ref
savepath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\rad760_year_comparison'
path_ref = r'E:\Datahub\Barbeau\Data_SIF\Califiles\Reflectance References\Reflectance References Interpolation'
refl_refs = pd.read_csv(os.path.join(path_ref, 'All_Interpolated_Reflectance.csv'))
# refl_refs = pd.read_csv(os.path.join(path_ref, 'All_Interpolated_Reflectance_usestart.csv'))
fig, ax = plt.subplots(figsize=(12, 8))
wl = refl_refs.columns[1:].astype(float)
ax.imshow(refl_refs.iloc[:, 1:].T, aspect='auto', cmap='viridis', extent=[0, refl_refs.shape[0], refl_refs.shape[1]-1, 0], vmin = 90, vmax = 100)
cbar = fig.colorbar(ax.imshow(refl_refs.iloc[:, 1:].T, aspect='auto', cmap='viridis', extent=[0, refl_refs.shape[0], refl_refs.shape[1]-1, 0], vmin = 90, vmax = 100), ax=ax)
cbar.set_label('Reflectance of panel (%)', rotation=270, labelpad=15)
ax.set_xlabel('Index')
ax.set_xticks(np.arange(0, refl_refs.shape[0], 50))
ax.set_xticklabels(refl_refs['date'].iloc[::50], rotation=45)
ax.set_yticks(np.arange(0, refl_refs.shape[1]-1, 50))
ax.set_yticklabels(wl[::50])
ax.set_ylabel('Wavelength (nm)')
ax.set_title('Reflectance References Interpolation')
plt.tight_layout()
plt.savefig(os.path.join(savepath, 'Reflectance_References_Interpolation.png'), dpi=300, bbox_inches='tight')
# plt.savefig(os.path.join(savepath, 'Reflectance_References_Interpolation_usestart.png'), dpi=300, bbox_inches='tight')


# %% 
datapath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
years = ['2022','2023','2024','2025']

# %% DN check
# fig, ax = plt.subplots(4, 1, figsize=(15, 10))
# fig1, ax1 = plt.subplots(4, 1, figsize=(15, 10))
# fig2, ax2 = plt.subplots(4, 1, figsize=(15, 10))
# for iyear, year in enumerate(years):
#     print(f'Processing year: {year}')
#     yearpath = os.path.join(datapath, year,'PROCESSED\\L2\\Yearly')
#     df_sif = pd.read_csv(os.path.join(yearpath, 'PROSIF_rad_irrad_760.csv'), index_col=False)
#     df_sif.loc[(df_sif['rad_760'] < 0) | (df_sif['rad_760'] > 20000), 'rad_760'] = np.nan
#     df_sif.loc[(df_sif['irrad_760'] < 0) | (df_sif['irrad_760'] > 20000), 'irrad_760'] = np.nan
#     # time_spend
#     df_sif['time_spend'] = (df_sif['Time_end'] - df_sif['Time_start']) * 24 * 60 * 60 # seconds
#     # idx_time = (df_sif['time_spend'] > 1) & (df_sif['time_spend'] <5.5)
#     # idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
#     # df_sif = df_sif[idx_time & idx_value]

#     # idx_time = (df_sif['time_spend'] >= 0) & (df_sif['time_spend'] <5.5)
#     # idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
#     # df_sif = df_sif[idx_value & idx_time]

#     df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
#     df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])

#     # plot doy vs SIF_3FLD
#     sc = ax[iyear].scatter(df_sif['DOY_sif'], df_sif['rad_760'], s=4, c=df_sif['PAR_micromoles_m_2s_1'], cmap='viridis', label='SIF_3FLD', vmin = 0, vmax = 2000)
#     fig.colorbar(sc, ax=ax[iyear], label='PAR_micromoles_m_2s_1')
#     if iyear == 3:
#         ax[iyear].set_xlabel('Day of Year')
#     ax[iyear].set_ylabel('(DN-DC)/inttime')
#     ax[iyear].set_title(f'Year {year}')
#     ax[iyear].grid()
#     ax[iyear].set_xlim(100, 300)
#     ax[iyear].set_ylim(0, 21000)
#     # ax[iyear].legend()
#     # plot doy vs irrad_760
#     sc1 = ax1[iyear].scatter(df_sif['DOY_sif'], df_sif['irrad_760'], s=4, c=df_sif['PAR_micromoles_m_2s_1'], cmap='viridis', label='SIF_3FLD', vmin = 0, vmax = 2000)
#     fig1.colorbar(sc1, ax=ax1[iyear], label='PAR_micromoles_m_2s_1')
#     if iyear == 3:
#         ax1[iyear].set_xlabel('Day of Year')
#     ax1[iyear].set_ylabel('(DN-DC)/inttime')
#     ax1[iyear].set_title(f'Year {year}')
#     ax1[iyear].grid()
#     ax1[iyear].set_xlim(100, 300)
#     ax1[iyear].set_ylim(0, 21000)

#     # plot doy vs PAR
#     sc2 = ax2[iyear].scatter(df_sif['DOY_sif'], df_sif['PAR_micromoles_m_2s_1'], s=4, c=df_sif['PAR_micromoles_m_2s_1'], cmap='viridis', label='SIF_3FLD', vmin = 0, vmax = 2000)
#     fig2.colorbar(sc2, ax=ax2[iyear], label='PAR_micromoles_m_2s_1')
#     if iyear == 3:
#         ax2[iyear].set_xlabel('Day of Year')    
#     ax2[iyear].set_ylabel('PAR (micromoles m-2 s-1)')
#     ax2[iyear].set_title(f'Year {year}')
#     ax2[iyear].grid()
#     ax2[iyear].set_xlim(100, 300)
#     ax2[iyear].set_ylim(0, 2400)

# fig.savefig(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\rad_check.png', dpi=300)
# fig1.savefig(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\irrad_check.png', dpi=300)
# fig2.savefig(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\PAR_check.png', dpi=300)
# %% rad check
# fig, ax = plt.subplots(4, 1, figsize=(15, 10))
# fig1, ax1 = plt.subplots(4, 1, figsize=(15, 10))
# for iyear, year in enumerate(years):
#     print(f'Processing year: {year}')
#     yearpath = os.path.join(datapath, year,'PROCESSED\\L2\\Yearly')
#     df_sif = pd.read_csv(os.path.join(yearpath, 'PROSIF_SIFresults_' + year + '_Yearly.csv'), index_col=False)

#     # time_spend
#     df_sif['time_spend'] = (df_sif['Time_end'] - df_sif['Time_start']) * 24 * 60 * 60 # seconds
#     # idx_time = (df_sif['time_spend'] > 1) & (df_sif['time_spend'] <5.5)
#     # idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
#     # df_sif = df_sif[idx_time & idx_value]

#     idx_time = (df_sif['time_spend'] >= 0) & (df_sif['time_spend'] <5.5)
#     idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
#     df_sif = df_sif[idx_value & idx_time]

#     df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
#     df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])

#     # plot doy vs SIF_3FLD
#     sc = ax[iyear].scatter(df_sif['DOY_sif'], df_sif['rad_760'], s=4, c=df_sif['SIF_3FLD'], cmap='viridis', label='SIF_3FLD', vmin = 0, vmax = 3)
#     fig.colorbar(sc, ax=ax[iyear], label='SIF_3FLD')
#     if iyear == 3:
#         ax[iyear].set_xlabel('Day of Year')
#     ax[iyear].set_ylabel('Radiance@760nm'+'\n'+'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
#     ax[iyear].set_title(f'Year {year}')
#     ax[iyear].grid()
#     ax[iyear].set_xlim(100, 300)
#     ax[iyear].set_ylim(0, 60)
#     # ax[iyear].legend()
#     # plot doy vs irrad_760
#     sc1 = ax1[iyear].scatter(df_sif['DOY_sif'], df_sif['irrad_760'], s=4, c=df_sif['SIF_3FLD'], cmap='viridis', label='SIF_3FLD', vmin = 0, vmax = 3)
#     fig1.colorbar(sc1, ax=ax1[iyear], label='SIF_3FLD')
#     if iyear == 3:
#         ax1[iyear].set_xlabel('Day of Year')
#     ax1[iyear].set_ylabel('Irradiance@760nm'+'\n'+'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
#     ax1[iyear].set_title(f'Year {year}')
#     ax1[iyear].grid()
#     ax1[iyear].set_xlim(100, 300)
#     ax1[iyear].set_ylim(0, 500)

# fig.savefig(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\rad_check_rad.png', dpi=300)
# fig1.savefig(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\irrad_check_rad.png', dpi=300)
plt.show()