# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import datetime
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datapath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data'
years = ['2022','2023','2024','2025']

path_wl = r'E:\Datahub\Barbeau\Data_SIF\Califiles'
# wavelength data
wl_hr = pd.read_csv(os.path.join(path_wl, 'HR_WL.csv'))
# 找到wl_hr中距离760nm最近的波长索引
IDX_760 = np.argmin(np.abs(wl_hr['WL'].to_numpy() - 760)) # 取 idx_760 -1 : idx_760 + 1

for iyear, year in enumerate(years):
    print(f'Processing year: {year}')
    rawpath = os.path.join(datapath, year,'PROCESSED\\L1\\RAW')
    metapath = os.path.join(datapath, year,'PROCESSED\\L1\\META')
    df_rad = pd.DataFrame([], columns=['Time_start', 'Time_end', 'Time_mid', 'Integration_time','rad_760_dn','rad_760_dc','irrad_bf_760_dn','irrad_af_760_dn','irrad_bf_760_dc','irrad_af_760_dc','rad_760', 'irrad_760','PAR_micromoles_m_2s_1'])
    for csv_file in glob.glob(os.path.join(rawpath, '*HR.csv')):
        filename = os.path.basename(csv_file)
        df_raw = pd.read_csv(csv_file, index_col=False)
        df_meta = pd.read_csv(os.path.join(metapath, filename.replace('.csv', '_meta.csv')), index_col=False)
        rad_760 = np.nanmean(df_raw.iloc[IDX_760-1:IDX_760+2,:], axis=0) # average over 3 pixels around 760nm
        rad_760_dn = rad_760[0::6]
        irrad_bf_760_dn = rad_760[1::6]
        irrad_af_760_dn = rad_760[2::6]
        rad_760_dc = rad_760[3::6]
        irrad_bf_760_dc = rad_760[4::6]
        irrad_af_760_dc = rad_760[5::6]

        
        inttime = df_meta['Integration_time'].to_numpy()/1e6 # ms to s
        rad_760 = (rad_760_dn-rad_760_dc) / inttime
        irrad_bf_760 = (irrad_bf_760_dn - irrad_bf_760_dc) / inttime
        irrad_af_760 = (irrad_af_760_dn - irrad_af_760_dc) / inttime
        irrad_760 = (irrad_bf_760 + irrad_af_760) / 2 # average of before and after

        time_start_dt = df_meta['Time_start'] # time of rad_760[0::3]
        time_end_dt = df_meta['Time_end']
        time_mid_dt = time_start_dt + (time_end_dt - time_start_dt) / 2

        par = df_meta['PAR_micromoles_m_2s_1'] # PAR in micromoles/m^2/s

        df_rad = pd.concat([df_rad, pd.DataFrame({'Time_start': time_start_dt, 'Time_end': time_end_dt, 'Time_mid': time_mid_dt, 'Integration_time': inttime, 'rad_760_dn': rad_760_dn, 'rad_760_dc': rad_760_dc, 'irrad_bf_760_dn': irrad_bf_760_dn, 'irrad_af_760_dn': irrad_af_760_dn, 'irrad_bf_760_dc': irrad_bf_760_dc, 'irrad_af_760_dc': irrad_af_760_dc, 'rad_760': rad_760, 'irrad_760': irrad_760, 'PAR_micromoles_m_2s_1': par})], ignore_index=True)

    df_rad.to_csv(os.path.join(datapath, year,'PROCESSED\\L2\\Yearly\\PROSIF_rad_irrad_760.csv'), index=False)














# fig, ax = plt.subplots(4, 1, figsize=(18, 10))