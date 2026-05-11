# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def read_dx_file(file_path):
    wl_start = pd.read_csv(file_path, skiprows=24, nrows=1, sep=' ', 
                           names=['Wavelength', 'Reflectance'])['Wavelength']
    wl_start = int(wl_start.values[0]) if not wl_start.empty else None
    print(f"Reading file: {file_path}, starting wavelength: {wl_start}")
    data = pd.read_csv(file_path, skiprows=24, nrows = 1000 - wl_start + 1, sep = ' ',
                       names=['Wavelength', 'Reflectance'])
    # 对data进行SG滤波平滑
    data['Reflectance_sm'] = savgol_filter(data['Reflectance'], window_length=151, polyorder=3)

    return data

def comp_reflectance(file_path1, file_path2, fig_path=None):
    data_new = read_dx_file(file_path1)
    data_old = read_dx_file(file_path2)
    label1 = os.path.basename(file_path1)
    label2 = os.path.basename(file_path2)
    # # 确保两个数据集的波长轴一致
    # if not np.array_equal(data_new['Wavelength'], data_old['Wavelength']):
    #     raise ValueError("两个数据集的波长轴不一致")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(data_new['Wavelength'], data_new['Reflectance'], label=label1, color='blue')
    ax.plot(data_new['Wavelength'], data_new['Reflectance_sm'], label=label1 + ' (Smoothed)', color='cyan', linestyle='--')
    ax.plot(data_old['Wavelength'], data_old['Reflectance'], label=label2, color='orange')
    ax.plot(data_old['Wavelength'], data_old['Reflectance_sm'], label=label2 + ' (Smoothed)', color='red', linestyle='--')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Reflectance Comparison')
    ax.legend()
    if fig_path is not None:
        plt.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    root = r'E:\Datahub\Barbeau\Data_SIF\Califiles\Reflectance References'
    # 
    folder = '1_Reflectance References 24 02 2022'
    DX_file_new = os.path.join(root, folder, 'RTOT_SIF3_NEW_24 02 2022.DX')
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_OLD_24 02 2022.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder, 'Reflectance_Comparison.png'))
    # 
    folder1 = '2_Reflectance References 05 07 2022'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 2_05 07 2022.DX')
    folder = '1_Reflectance References 24 02 2022'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW_24 02 2022.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png'))
    # 
    folder = '3_Reflectance References 05 01 2023'
    DX_file_new = os.path.join(root, folder, 'RTOT_SIF3_NEW AFTER_05 01 2023.DX')
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW RED SPOT_05 01 2023.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder, 'Reflectance_Comparison.png'))
    # 
    folder1 = '4_Reflectance References 12 06 2023'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 3_12 06 2023.DX')
    folder = '3_Reflectance References 05 01 2023'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW AFTER_05 01 2023.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png'))
    # 
    folder1 = '5_Reflectance References 01 03 2024'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 2 AFTER_01 03 2024.DX')
    folder = '4_Reflectance References 12 06 2023'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 3_12 06 2023.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png'))
    # 
    folder1 = '6_Reflectance References 08 03 2024'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 4_08 03 2024.DX')
    folder = '5_Reflectance References 01 03 2024'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 2 AFTER_01 03 2024.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png'))   
    # 
    folder1 = '7_Reflectance References 11 03 2025'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 3 AFTER_11 03 2025.DX')
    folder = '6_Reflectance References 08 03 2024'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 4_08 03 2024.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png'))  
    # 
    folder1 = '8_Reflectance References 03 04 2025'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 5_03 04 2025.DX')
    folder = '7_Reflectance References 11 03 2025'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 3 AFTER_11 03 2025.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png')) 
    # 
    folder1 = '9_Reflectance References 16 04 2025'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 4 AFTER_16 04 2025.DX')
    folder = '8_Reflectance References 03 04 2025'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 5_03 04 2025.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png')) 
    # 
    folder1 = '10_Reflectance References 17 03 2026'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 6_17 03 2026.DX')
    folder = '9_Reflectance References 16 04 2025'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 4 AFTER_16 04 2025.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png')) 
    #
    folder1 = '11_Reflectance References 23 03 2026'
    DX_file_new = os.path.join(root, folder1, 'RTOT_SIF3_NEW 5 AFTER_23 03 2026.DX')
    folder = '10_Reflectance References 17 03 2026'
    DX_file_OLD = os.path.join(root, folder, 'RTOT_SIF3_NEW 6_17 03 2026.DX')
    comp_reflectance(DX_file_new, DX_file_OLD, fig_path=os.path.join(root, folder1, 'Reflectance_Comparison.png')) 


    print('a')