# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from scipy.signal import savgol_filter

def read_dx_file(file_path):
    wl_start = pd.read_csv(file_path, skiprows=24, nrows=1, sep=' ', 
                           names=['Wavelength', 'Reflectance'])['Wavelength']
    wl_start = int(wl_start.values[0]) if not wl_start.empty else None
    data = pd.read_csv(file_path, skiprows=24, nrows = 1000 - wl_start + 1, sep = ' ',
                       names=['Wavelength', 'Reflectance'])
    # 对data进行SG滤波平滑
    data['Reflectance_sm'] = savgol_filter(data['Reflectance'], window_length=151, polyorder=3)
    return data

def get_date_from_filename(file_path):
    date_str = os.path.basename(file_path).split('_')[-1].replace('.DX', '')
    day, mon, year = date_str.split(' ')
    datetime_obj = pd.to_datetime(f'{day} {mon} {year}', format='%d %m %Y')
    return datetime_obj

def get_all_dates_between(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    return date_range

def get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, save_path):
    data_ref_begin = read_dx_file(path_ref_begin)
    data_ref_end = read_dx_file(path_ref_end)
    # start_date = get_date_from_filename(path_ref_begin)
    # end_date = get_date_from_filename(path_ref_end)
    start_date = pd.to_datetime(dates_range[0])
    end_date = pd.to_datetime(dates_range[1])
    date_range = get_all_dates_between(start_date, end_date)

    wl_begin = data_ref_begin['Wavelength']
    wl_end = data_ref_end['Wavelength']
    if wl_begin[0] != wl_end[0] or wl_begin.iloc[-1] != wl_end.iloc[-1]:
        print('Warning: Wavelength ranges do not match between the two files.')
    if wl_begin[0] < wl_end[0]:
        wl_begin = wl_begin[wl_begin >= wl_end[0]].reset_index(drop=True)
        data_ref_begin = data_ref_begin.loc[data_ref_begin['Wavelength'] >= wl_end[0],:].reset_index(drop=True)
    elif wl_end[0] < wl_begin[0]:
        wl_end = wl_end[wl_end >= wl_begin[0]].reset_index(drop=True)
        data_ref_end = data_ref_end.loc[data_ref_end['Wavelength'] >= wl_begin[0],:].reset_index(drop=True)

    refl_begin = data_ref_begin['Reflectance_sm']
    refl_end = data_ref_end['Reflectance_sm']
    
    interpolated_refs = pd.DataFrame([],columns=['date']+list(wl_begin))
    for i, date in enumerate(date_range):
        alpha = (date - start_date).days / (end_date - start_date).days
        interpolated_refl = (1 - alpha) * refl_begin + alpha * refl_end
        interpolated_refs.loc[i, 'date'] = date
        interpolated_refs.iloc[i, 1:len(wl_begin)+1] = interpolated_refl.values
        
    # save interpolated reflectance curves
    interpolated_refs.to_csv(os.path.join(save_path, f'Interpolated_Reflectance_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'), index=False)
    # plot interpolated reflectance curves
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    for i in range(len(interpolated_refs)):
        ax.plot(interpolated_refs.columns[1:], interpolated_refs.iloc[i, 1:]) #, label=interpolated_refs.iloc[i, 0].strftime('%Y-%m-%d')
    ax.plot(data_ref_begin['Wavelength'], data_ref_begin['Reflectance_sm'], 
            label='Reference Begin ('+start_date.strftime("%Y-%m-%d")+')', color='blue', linestyle='--')    
    ax.plot(data_ref_end['Wavelength'], data_ref_end['Reflectance_sm'], 
            label='Reference End ('+end_date.strftime("%Y-%m-%d")+')', color='red', linestyle='--')    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Interpolated Reflectance' + f' from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'Interpolated_Reflectance_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.png'), dpi=300)

    return interpolated_refs

if __name__ == "__main__":
    root = r'E:\Datahub\Barbeau\Data_SIF\Califiles\Reflectance References'
    savepath = os.path.join(root, 'Reflectance References Interpolation')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # from 2022-02-24 to 2022-07-05
    path_ref_begin = os.path.join(root, '1_Reflectance References 24 02 2022','RTOT_SIF3_NEW_24 02 2022.DX')
    path_ref_end = os.path.join(root, '3_Reflectance References 05 01 2023','RTOT_SIF3_NEW AFTER_05 01 2023.DX')
    dates_range = [datetime.date(2022, 2, 24), datetime.date(2022, 7, 7)]
    interpolated_refs1 = get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, savepath)
    # 2022-07-05 to 2023-06-12
    path_ref_begin = os.path.join(root, '2_Reflectance References 05 07 2022','RTOT_SIF3_NEW 2_05 07 2022.DX')
    path_ref_end = os.path.join(root, '5_Reflectance References 01 03 2024','RTOT_SIF3_NEW 2 AFTER_01 03 2024.DX')
    dates_range = [datetime.date(2022, 7, 7), datetime.date(2023, 6, 15)]
    interpolated_refs2 = get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, savepath)
    # 2023-06-12 to 2024-03-08
    path_ref_begin = os.path.join(root, '4_Reflectance References 12 06 2023','RTOT_SIF3_NEW 3_12 06 2023.DX')
    path_ref_end = os.path.join(root, '7_Reflectance References 11 03 2025','RTOT_SIF3_NEW 3 AFTER_11 03 2025.DX')
    dates_range = [datetime.date(2023, 6, 15), datetime.date(2024, 3, 19)]
    interpolated_refs3 = get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, savepath)
    # 2024-03-08 to 2025-04-03
    path_ref_begin = os.path.join(root, '6_Reflectance References 08 03 2024','RTOT_SIF3_NEW 4_08 03 2024.DX')
    path_ref_end = os.path.join(root, '9_Reflectance References 16 04 2025','RTOT_SIF3_NEW 4 AFTER_16 04 2025.DX')
    dates_range = [datetime.date(2024, 3, 19), datetime.date(2025, 4, 7)]
    interpolated_refs4 = get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, savepath)
    # 2025-04-03 to 2026-03-17
    path_ref_begin = os.path.join(root, '8_Reflectance References 03 04 2025','RTOT_SIF3_NEW 5_03 04 2025.DX')
    path_ref_end = os.path.join(root, '11_Reflectance References 23 03 2026','RTOT_SIF3_NEW 5 AFTER_23 03 2026.DX')
    dates_range = [datetime.date(2025, 4, 7), datetime.date(2026, 3, 19)]
    interpolated_refs5 = get_interpolated_reflectance(path_ref_begin, path_ref_end, dates_range, savepath)

    ## remove repetitive interpolated reflectance curves
    # remove the first and the last rows of interpolated_refs2 to avoid duplication with interpolated_refs1 and interpolated_refs3
    interpolated_refs2 = interpolated_refs2.iloc[1:-1, :].reset_index(drop=True)
    # remove the the last rows of interpolated_refs3 to avoid duplication with interpolated_refs4
    interpolated_refs3 = interpolated_refs3.iloc[:-1, :].reset_index(drop=True)
    # remove the the last rows of interpolated_refs4 to avoid duplication with interpolated_refs5
    interpolated_refs4 = interpolated_refs4.iloc[:-1, :].reset_index(drop=True)
    # remove the the last rows of interpolated_refs5 to avoid duplication with interpolated_refs6(future)
    interpolated_refs5 = interpolated_refs5.iloc[:-1, :].reset_index(drop=True)
    # concatenate all interpolated reflectance curves    
    all_interpolated_refs = pd.concat([interpolated_refs1, interpolated_refs2, 
                                       interpolated_refs3, interpolated_refs4, interpolated_refs5])
    all_interpolated_refs.to_csv(os.path.join(savepath, 'All_Interpolated_Reflectance.csv'), index=False)