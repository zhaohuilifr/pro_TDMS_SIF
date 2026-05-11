# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
from scipy.interpolate import CubicSpline
import scipy.io
import readTDMS


def calculate_refl(dat, meta, target_refl, int_time_col, r1_int_col, r2_int_col):
    # 提取信号与暗电流 (S_net = S_signal - S_dark)
    s_net = dat[:, :, 0, 0] - dat[:, :, 1, 0]
    r1_net = dat[:, :, 0, 1] - dat[:, :, 1, 1]
    r2_net = dat[:, :, 0, 2] - dat[:, :, 1, 2]
    
    # 积分时间比率
    ratio_i1 = (meta[int_time_col] / meta[r1_int_col]).values
    ratio_i2 = (meta[int_time_col] / meta[r2_int_col]).values
    
    # 分别基于前参考(R1)和后参考(R2)计算反射率
    # np.newaxis 用于对齐波长维度
    refl_i1 = (target_refl[:, np.newaxis] * (s_net / r1_net)) / ratio_i1
    refl_i2 = (target_refl[:, np.newaxis] * (s_net / r2_net)) / ratio_i2
    
    # 结果堆叠为 3D: [pixels, groups, 2(基于R1/基于R2)]
    return np.stack([refl_i1, refl_i2], axis=2)

def compute_bands(refl_data, wl_axis, centers, bandwidth):
    results = np.zeros((len(centers), refl_data.shape[1]))
    for i, center in enumerate(centers):
        # 找到带宽范围内的波长索引
        mask = (wl_axis >= (center - bandwidth/2)) & (wl_axis <= (center + bandwidth/2))
        if mask.any():
            results[i, :] = np.nanmean(refl_data[mask, :], axis=0)
    return results

def calculate_rad(dat, meta, coef, int_col, r1_int_col, r2_int_col):
    # 提取净计数
    s_net = dat[:, :, 0, 0] - dat[:, :, 1, 0]
    r1_net = dat[:, :, 0, 1] - dat[:, :, 1, 1]
    r2_net = dat[:, :, 0, 2] - dat[:, :, 1, 2]
    
    # 应用定标系数 (COEF) 并除以积分时间
    rad_s = (coef[:, np.newaxis] * s_net) / meta[int_col].values
    rad_r1 = (coef[:, np.newaxis] * r1_net) / meta[r1_int_col].values
    rad_r2 = (coef[:, np.newaxis] * r2_net) / meta[r2_int_col].values
    
    return np.stack([rad_s, rad_r1, rad_r2], axis=2)

def mat_metadata_to_df(mat_metadata):
    """
    将 loadmat 加载的 MATLAB 结构体数组转换为 pandas DataFrame
    """
    # MATLAB 结构体通常以 (1, N) 或 (N, 1) 的形状加载
    records = mat_metadata[0] if mat_metadata.shape[0] == 1 else mat_metadata[:, 0]
    
    # 获取第一个记录的 dtype 来提取字段名 (fields)
    # 在 L0 文件中，每个记录通常是一个包含 1x1 结构体的嵌套 array
    first_record_structured = records[0][0, 0]
    field_names = first_record_structured.dtype.names
    
    data_list = []
    for rec in records:
        # 提取内部的结构化标量对象
        obj = rec[0, 0]
        row = {}
        for name in field_names:
            # print(name)
            val = obj[name]
            # 循环解包，直到获取到基本数值/字符串 (处理如 array([[val]]) 的情况)
            while isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            row[name] = val
        data_list.append(row)
    
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    Year = 2025 #2023 # 2024, 2025
    path = os.path.join(r'E:\Datahub\Barbeau\Data_SIF\SIF3data', str(Year))
    pathcalib = r'E:\Datahub\Barbeau\Data_SIF\Califiles'
    path_ref = r'E:\Datahub\Barbeau\Data_SIF\Califiles\Reflectance References\Reflectance References Interpolation'
    # 1. 加载元数据
    metaI = pd.read_excel(os.path.join(pathcalib, 'instrument.xlsx'))
    metaI = metaI.loc[metaI['year']==Year, :].reset_index(drop=True)
    ccalib = {
        'LR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['LR_WL'].values,
        'LR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['LR_COEF'].values,
        'HR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['HR_WL'].values,
        'HR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['HR_COEF'].values
    }
    # refl_ref = pd.read_excel(os.path.join(pathcalib, 'PMR10P1.xlsx'))
    refl_refs = pd.read_csv(os.path.join(path_ref, 'All_Interpolated_Reflectance.csv'))
    refl_refs['date'] = pd.to_datetime(refl_refs['date'])
    # 2. 创建输出目录
    for level in ['L0', 'L1', 'L2', 'L1/RAW', 'L1/REFL', 'L1/BANDS/LR', 'L1/CAL']:
        os.makedirs(os.path.join(path, 'PROCESSED', level), exist_ok=True)

    # # 3. 原始文件处理 (L0)
    # tdms_files = glob.glob(os.path.join(path, "RAW", '*.tdms'))
    # for i, tdms_file in enumerate(tdms_files):
    #     print(f"Processing file {i+1}/{len(tdms_files)}: {tdms_file}")
    #     fnname = os.path.splitext(os.path.basename(tdms_file))[0]
    #     out_l0 = os.path.join(path, 'PROCESSED', 'L0', f'{fnname}.mat')
    #     # readTDMS.read_sif_file(path, fnname, metaI, ccalib)
    #     if not os.path.exists(out_l0):
    #         readTDMS.read_sif_file(path, fnname, metaI, ccalib)
    
    # %%  4. L0 数据整合与 L1 计算 (以 site_index > 0 的复杂逻辑为例)
    l0_files = glob.glob(os.path.join(path, 'PROCESSED', 'L0', '*.mat'))
    all_data_hr = []
    all_meta_hr = pd.DataFrame()
    all_data_lr = []
    all_meta_lr = pd.DataFrame()
    
    for f in l0_files:
        # 获取文件名中的日期，用于后续的白板反射率数据的时间匹配，从而得到对应的反射率定标系数
        date_str = pd.to_datetime((os.path.basename(f)).split('.')[0])
        tmp = refl_refs.iloc[(refl_refs['date'] - date_str).abs().argsort()[:1]] # 获取最接近的反射率参考数据
        refl_ref = pd.DataFrame([], columns=['WL', 'REFL'])
        refl_ref['WL'] = tmp.columns[1:].astype(float)
        refl_ref['REFL'] = tmp.iloc[0, 1:].values.astype(float)
        # 加载 L0 数据
        data = scipy.io.loadmat(f)
        #----------------- 处理 HR 数据 -----------------
        m_hr_s = mat_metadata_to_df(data['meta_HR_S'])
        m_hr_r = mat_metadata_to_df(data['meta_HR_R'])
        # 时间匹配逻辑: 寻找最接近的 Reference 帧
        l_before = []
        l_after = []
        for _, s_row in m_hr_s.iterrows():
            # 简化逻辑：寻找在样本开始前结束的最接近的 R
            diff_before = s_row['Time_start'] - m_hr_r['Time_end']
            diff_before[diff_before < 0] = np.nan
            # 寻找在样本结束后开始的最接近的 R
            diff_after = m_hr_r['Time_start'] - s_row['Time_end']
            diff_after[diff_after < 0] = np.nan

            idx_b = diff_before.idxmin() if not diff_before.isna().all() else None
            idx_a = diff_after.idxmin() if not diff_after.isna().all() else None
            
            # 如果找不到 before，则用 after 代替，反之亦然 (对应 MATLAB 的 isempty 判断)
            final_b = idx_b if idx_b is not None else idx_a
            final_a = idx_a if idx_a is not None else idx_b
            
            l_before.append(final_b)
            l_after.append(final_a)
        # 将匹配到的 Reference 元数据合并到 Sample 元数据中 (I1=before, I2=after)
        for col in m_hr_r.columns:
            if col not in ['idx']: # 排除不需要重复的列
                m_hr_s[f'I1_{col}'] = m_hr_r[col].iloc[l_before].values
                m_hr_s[f'I2_{col}'] = m_hr_r[col].iloc[l_after].values
        # 拼接 4D 数据: [像素, 样本组, (Signal/Dark), (S/R_before/R_after)]
        # a['HR_S'] 维度通常是 (pixels, groups, 2)
        s_dat = data['HR_S']
        r_before = data['HR_R'][:, l_before, :]
        r_after = data['HR_R'][:, l_after, :]
        # 合并为一个 4 维数组 (对应 MATLAB 的 cat(4, ...))
        combined_hr = np.stack([s_dat, r_before, r_after], axis=3)
        all_data_hr.append(combined_hr)
        all_meta_hr = pd.concat([all_meta_hr, m_hr_s], ignore_index=True)
        #----------------- 类似地处理 LR 数据 -----------------
        m_lr_s = mat_metadata_to_df(data['meta_LR_S'])
        m_lr_r = mat_metadata_to_df(data['meta_LR_R'])
        l_before_lr = []
        l_after_lr = []
        for _, s_row in m_lr_s.iterrows():
            diff_before = s_row['Time_start'] - m_lr_r['Time_end']
            diff_before[diff_before < 0] = np.nan
            diff_after = m_lr_r['Time_start'] - s_row['Time_end']
            diff_after[diff_after < 0] = np.nan

            idx_b = diff_before.idxmin() if not diff_before.isna().all() else None
            idx_a = diff_after.idxmin() if not diff_after.isna().all() else None
            
            final_b = idx_b if idx_b is not None else idx_a
            final_a = idx_a if idx_a is not None else idx_b
            
            l_before_lr.append(final_b)
            l_after_lr.append(final_a)
        for col in m_lr_r.columns:
            if col not in ['idx']:
                m_lr_s[f'I1_{col}'] = m_lr_r[col].iloc[l_before_lr].values
                m_lr_s[f'I2_{col}'] = m_lr_r[col].iloc[l_after_lr].values
        s_dat_lr = data['LR_S']
        r_before_lr = data['LR_R'][:, l_before_lr, :]
        r_after_lr = data['LR_R'][:, l_after_lr, :]
        combined_lr = np.stack([s_dat_lr, r_before_lr, r_after_lr], axis=3)
        all_data_lr.append(combined_lr)
        all_meta_lr = pd.concat([all_meta_lr, m_lr_s], ignore_index=True)

    # 最终合并所有文件的数据
    final_data_hr = np.concatenate(all_data_hr, axis=1) # 在样本组维度上合并
    final_data_lr = np.concatenate(all_data_lr, axis=1)

    # --- 暗电流修正 (对应 MATLAB 的 correction for missing or bad dark noise) ---
    # 如果暗电流坏点比例 > 2.5%，则使用该帧暗电流的平均值填充整个光谱 (解决异常值)
    bad_mask = all_meta_hr['prcdarkpix_sig'] > 0.025
    if bad_mask.any():
        # axis 0: pixels, axis 2: 1 (Dark), axis 3: 0 (Sample)
        final_data_hr[:, bad_mask, 1, 0] = all_meta_hr.loc[bad_mask, 'dark_dpix_mean'].values
    # 同理修正 I1 (Reference before) 和 I2 (Reference after) 的暗电流
    bad_mask_i1 = all_meta_hr['I1_prcdarkpix_sig'] > 0.025
    if bad_mask_i1.any():
        final_data_hr[:, bad_mask_i1, 1, 1] = all_meta_hr.loc[bad_mask_i1, 'I1_dark_dpix_mean'].values
    bad_mask_i2 = all_meta_hr['I2_prcdarkpix_sig'] > 0.025
    if bad_mask_i2.any():
        final_data_hr[:, bad_mask_i2, 1, 2] = all_meta_hr.loc[bad_mask_i2, 'I2_dark_dpix_mean'].values

     # 对 LR 数据同样处理
    bad_mask_lr = all_meta_lr['prcdarkpix_sig'] > 0.025
    if bad_mask_lr.any():
        final_data_lr[:, bad_mask_lr, 1, 0] = all_meta_lr.loc[bad_mask_lr, 'dark_dpix_mean'].values
    bad_mask_i1_lr = all_meta_lr['I1_prcdarkpix_sig'] > 0.025
    if bad_mask_i1_lr.any():
        final_data_lr[:, bad_mask_i1_lr, 1, 1] = all_meta_lr.loc[bad_mask_i1_lr, 'I1_dark_dpix_mean'].values
    bad_mask_i2_lr = all_meta_lr['I2_prcdarkpix_sig'] > 0.025
    if bad_mask_i2_lr.any():
        final_data_lr[:, bad_mask_i2_lr, 1, 2] = all_meta_lr.loc[bad_mask_i2_lr, 'I2_dark_dpix_mean'].values
    # 保存最终的 L1 数据
    savemat(os.path.join(path, 'PROCESSED', 'L1', 'L1_HR.mat'), {
        'data_HR': final_data_hr,
        'meta_HR': all_meta_hr
    })
    savemat(os.path.join(path, 'PROCESSED', 'L1', 'L1_LR.mat'), {
        'data_LR': final_data_lr,
        'meta_LR': all_meta_lr
    })

    # %% 5. 计算反射率，radiance
    # --- 1. 参考板反射率插值 ---
    spl_ref = CubicSpline(refl_ref['WL'], refl_ref['REFL'] / 100.0)
    hr_target_refl = spl_ref(ccalib['HR_WL']) # 形状: (pixels,)
    lr_target_refl = spl_ref(ccalib['LR_WL'])
    # --- 2. 反射率计算 (Reflectance) ---
    # 计算公式: Refl = (Target_Refl) * (S_net / R_net) * (R_int_time / S_int_time)
    # dat_hr 维度: [pixels, groups, 2(Sig/Dark), 3(S/R1/R2)]
    hr_refl = calculate_refl(final_data_hr, all_meta_hr, hr_target_refl, 
                             'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
    lr_refl = calculate_refl(final_data_lr, all_meta_lr, lr_target_refl, 
                             'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
    # --- 3. 波段聚合 (Bands Aggregation) ---
    # 计算 10nm 和 25nm 带宽的 LR 数据
    band_centers = np.arange(400, 905, 5)
    bands_10nm = compute_bands(lr_refl[:, :, 0], ccalib['LR_WL'], band_centers, 10)
    bands_25nm = compute_bands(lr_refl[:, :, 0], ccalib['LR_WL'], band_centers, 25)
    # --- 4. 辐射定标 (Radiance) ---
    # 计算公式: Rad = COEF * (Net_Counts / Integration_Time)
    hr_cal = calculate_rad(final_data_hr, all_meta_hr, ccalib['HR_COEF'], 
                           'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
    lr_cal = calculate_rad(final_data_lr, all_meta_lr, ccalib['LR_COEF'], 
                           'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
    # --- 5. 保存 L1 级数据 ---
    refl_save_path = os.path.join(path, 'PROCESSED', 'L1', 'REFL', f'REFL.mat')
    savemat(refl_save_path, {
        'HR_Refl': hr_refl, 'LR_Refl': lr_refl,
        'meta_HR': all_meta_hr.to_dict('records'), 'meta_LR': all_meta_lr.to_dict('records')
    })
    cal_save_path = os.path.join(path, 'PROCESSED', 'L1', 'CAL', f'CAL.mat')
    savemat(cal_save_path, {
        'HR_Cal': hr_cal, 'LR_Cal': lr_cal,
        'meta_HR': all_meta_hr.to_dict('records'), 'meta_LR': all_meta_lr.to_dict('records')
    })