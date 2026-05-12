# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
# import shutil
import numpy as np
import pandas as pd
# from scipy.io import savemat, loadmat
from scipy.interpolate import CubicSpline
import scipy.io
import readTDMS


def match_reference_frames(sample_meta, reference_meta):
    """Match each sample row to the nearest before/after reference rows.

    Returns two index lists with the same semantics as the original loop:
    - before: latest reference whose Time_end is <= sample Time_start, falling back to after
    - after: earliest reference whose Time_start is >= sample Time_end, falling back to before
    """
    if reference_meta.empty:
        raise ValueError('reference_meta must not be empty for time matching')

    ref_start = (reference_meta['Time_start']).values
    ref_end = (reference_meta['Time_end']).values
    sample_start = (sample_meta['Time_start']).values
    sample_end = (sample_meta['Time_end']).values

    # ref_start = pd.to_datetime(reference_meta['Time_start']).to_numpy(dtype='datetime64[ns]')
    # ref_end = pd.to_datetime(reference_meta['Time_end']).to_numpy(dtype='datetime64[ns]')
    # sample_start = pd.to_datetime(sample_meta['Time_start']).to_numpy(dtype='datetime64[ns]')
    # sample_end = pd.to_datetime(sample_meta['Time_end']).to_numpy(dtype='datetime64[ns]')

    end_order = np.argsort(ref_end)
    start_order = np.argsort(ref_start)
    ref_end_sorted = ref_end[end_order]
    ref_start_sorted = ref_start[start_order]

    before_pos = np.searchsorted(ref_end_sorted, sample_start, side='right') - 1
    after_pos = np.searchsorted(ref_start_sorted, sample_end, side='left')
    
    # 安全方式：先创建结果数组，再根据条件填充
    before_idx = np.full(len(before_pos), -1, dtype=int)
    valid_before = before_pos >= 0
    before_idx[valid_before] = end_order[before_pos[valid_before]]

    after_idx = np.full(len(after_pos), -1, dtype=int)
    valid_after = after_pos < len(start_order)  # 关键：检查索引是否有效
    after_idx[valid_after] = start_order[after_pos[valid_after]]

    # before_idx = np.where(before_pos >= 0, end_order[before_pos], -1)
    # after_idx = np.where(after_pos < len(start_order), start_order[after_pos], -1)

    final_before = np.where(before_idx >= 0, before_idx, after_idx)
    final_after = np.where(after_idx >= 0, after_idx, before_idx)

    return final_before.tolist(), final_after.tolist()

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
    first_record_structured = records[0]
    field_names = first_record_structured.dtype.names
    
    data_list = []
    for rec in records:
        # 提取内部的结构化标量对象
        obj = rec
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
    pathcalib = r'E:\Datahub\Barbeau\Data_SIF\Califiles'
    path_ref = r'E:\Datahub\Barbeau\Data_SIF\Califiles\Reflectance References\Reflectance References Interpolation'
    # %% ---------------------------- 数据文件路径设置 ------------------ ------------------ #  
    # %% 2023, 2024, 2025 数据处理主程序
    # # 1. 加载元数据
    # Year = 2024 # 2023 # 2024, 2025
    # path = os.path.join(r'E:\Datahub\Barbeau\Data_SIF\SIF3data', str(Year))
    # metaI = pd.read_excel(os.path.join(pathcalib, 'instrument.xlsx'))
    # metaI = metaI.loc[metaI['year']==Year, :].reset_index(drop=True)
    # ccalib = {
    #     'LR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['LR_WL'].values,
    #     'LR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['LR_COEF'].values,
    #     'HR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['HR_WL'].values,
    #     'HR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2023.xlsx'))['HR_COEF'].values
    # }

    # %% 2022 
    # 1. 加载元数据
    path = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022'
    metaI = pd.read_excel(os.path.join(pathcalib, 'instrument.xlsx'))
    metaI = metaI.loc[metaI['year']==2022, :].reset_index(drop=True)
    ccalib = {
        'LR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2022.xlsx'))['LR_WL'].values,
        'LR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2022.xlsx'))['LR_COEF'].values,
        'HR_WL': pd.read_excel(os.path.join(pathcalib, 'calibration2022.xlsx'))['HR_WL'].values,
        'HR_COEF': pd.read_excel(os.path.join(pathcalib, 'calibration2022.xlsx'))['HR_COEF'].values
    }

    # %% ------------------------------------ 主程序核心逻辑 ------------------ ------------------ #
    # 反射率定标系数 
    # refl_refs = pd.read_csv(os.path.join(path_ref, 'All_Interpolated_Reflectance.csv'))
    # refl_refs['date'] = pd.to_datetime(refl_refs['date'])
    refl_ref = pd.read_excel(os.path.join(pathcalib, 'PMR10P1.xlsx'))

    # 2. 创建输出目录
    for level in ['L0', 'L1', 'L2', 'L1/RAW', 'L1/META','L1/REFL', 'L1/BANDS', 'L1/CAL']:
        os.makedirs(os.path.join(path, 'PROCESSED', level), exist_ok=True)

    # # 3. 原始文件处理 (L0)
    # tdms_files = glob.glob(os.path.join(path, "RAW", '*.tdms'))
    # for i, tdms_file in enumerate(tdms_files):
    #     print(f"Processing file {i+1}/{len(tdms_files)}: {tdms_file}")
    #     fnname = os.path.splitext(os.path.basename(tdms_file))[0]
    #     out_l0 = os.path.join(path, 'PROCESSED', 'L0', f'{fnname}.mat')
    #     # if not os.path.exists(out_l0):
    #     readTDMS.read_sif_file(path, fnname, metaI, ccalib)

    # %% 保存波长数据 (只需保存一次)
    wl_hr_df = pd.DataFrame(ccalib['HR_WL'], columns=['WL'])
    wl_hr_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','BANDS', 'HR_WL.csv'), index=False)
    wl_lr_df = pd.DataFrame(ccalib['LR_WL'], columns=['WL'])
    wl_lr_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','BANDS', 'LR_WL.csv'), index=False)

    # %%  4. L0 数据整合与 L1 计算 (以 site_index > 0 的复杂逻辑为例)
    l0_files = glob.glob(os.path.join(path, 'PROCESSED', 'L0', '*.mat'))
    all_data_hr = []
    all_meta_hr = pd.DataFrame()
    all_data_lr = []
    all_meta_lr = pd.DataFrame()
    
    for ifx, f in enumerate(l0_files):
        # 获取文件名中的日期，用于后续的白板反射率数据的时间匹配，从而得到对应的反射率定标系数
        date_str = pd.to_datetime((os.path.basename(f)).split('.')[0])

        # tmp = refl_refs.iloc[(refl_refs['date'] - date_str).abs().argsort()[:1]] # 获取最接近的反射率参考数据
        # refl_ref = pd.DataFrame([], columns=['WL', 'REFL'])
        # refl_ref['WL'] = tmp.columns[1:].astype(float)
        # refl_ref['REFL'] = tmp.iloc[0, 1:].values.astype(float)

        # 加载 L0 数据
        data = scipy.io.loadmat(f)
        #----------------- 处理 HR 数据 -----------------
        m_hr_s = mat_metadata_to_df(data['meta_HR_S'])
        m_hr_r = mat_metadata_to_df(data['meta_HR_R'])
        l_before, l_after = match_reference_frames(m_hr_s, m_hr_r)
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
        # 保存成csv文件
        csv_name = os.path.splitext(os.path.basename(f))[0] + '_HR_meta.csv'
        m_hr_s.to_csv(os.path.join(path, 'PROCESSED', 'L1','META', csv_name), index=False, columns=None)
        csv_name_hr = os.path.splitext(os.path.basename(f))[0] + '_HR.csv'
        combined_hr_df = pd.DataFrame(combined_hr.reshape(combined_hr.shape[0], -1))
        combined_hr_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','RAW', csv_name_hr), index=False, columns=None)

        #----------------- 类似地处理 LR 数据 -----------------
        m_lr_s = mat_metadata_to_df(data['meta_LR_S'])
        m_lr_r = mat_metadata_to_df(data['meta_LR_R'])
        l_before_lr, l_after_lr = match_reference_frames(m_lr_s, m_lr_r)
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
        # 保存成csv文件
        csv_name_lr = os.path.splitext(os.path.basename(f))[0] + '_LR_meta.csv'
        m_lr_s.to_csv(os.path.join(path, 'PROCESSED', 'L1','META', csv_name_lr), index=False, columns=None)
        csv_name_lr_dat = os.path.splitext(os.path.basename(f))[0] + '_LR.csv'
        combined_lr_df = pd.DataFrame(combined_lr.reshape(combined_lr.shape[0], -1))
        combined_lr_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','RAW', csv_name_lr_dat), index=False, columns=None)


        # # --- 暗电流修正 (对应 MATLAB 的 correction for missing or bad dark noise) ---
        # 如果暗电流坏点比例 > 2.5%，则使用该帧暗电流的平均值填充整个光谱 (解决异常值)
        bad_mask = m_hr_s['prcdarkpix_sig'] > 0.025
        if bad_mask.any():
            combined_hr[:, bad_mask, 1, 0] = m_hr_s.loc[bad_mask, 'dark_dpix_mean'].values
        # 同理修正 I1 (Reference before) 和 I2 (Reference after) 的暗电流
        bad_mask_i1 = m_hr_s['I1_prcdarkpix_sig'] > 0.025
        if bad_mask_i1.any():
            combined_hr[:, bad_mask_i1, 1, 1] = m_hr_s.loc[bad_mask_i1, 'I1_dark_dpix_mean'].values
        bad_mask_i2 = m_hr_s['I2_prcdarkpix_sig'] > 0.025
        if bad_mask_i2.any():
            combined_hr[:, bad_mask_i2, 1, 2] = m_hr_s.loc[bad_mask_i2, 'I2_dark_dpix_mean'].values

        # # 对 LR 数据同样处理
        bad_mask_lr = m_lr_s['prcdarkpix_sig'] > 0.025
        if bad_mask_lr.any():
            combined_lr[:, bad_mask_lr, 1, 0] = m_lr_s.loc[bad_mask_lr, 'dark_dpix_mean'].values
        bad_mask_i1_lr = m_lr_s['I1_prcdarkpix_sig'] > 0.025
        if bad_mask_i1_lr.any():
            combined_lr[:, bad_mask_i1_lr, 1, 1] = m_lr_s.loc[bad_mask_i1_lr, 'I1_dark_dpix_mean'].values
        bad_mask_i2_lr = m_lr_s['I2_prcdarkpix_sig'] > 0.025
        if bad_mask_i2_lr.any():
            combined_lr[:, bad_mask_i2_lr, 1, 2] = m_lr_s.loc[bad_mask_i2_lr, 'I2_dark_dpix_mean'].values

        # %% 5. 计算反射率，radiance
        # --- 1. 参考板反射率插值 ---
        spl_ref = CubicSpline(refl_ref['WL'], refl_ref['REFL'] / 100.0)
        hr_target_refl = spl_ref(ccalib['HR_WL']) # 形状: (pixels,)
        lr_target_refl = spl_ref(ccalib['LR_WL'])
        # --- 2. 反射率计算 (Reflectance) ---
        # 计算公式: Refl = (Target_Refl) * (S_net / R_net) * (R_int_time / S_int_time)
        # dat_hr 维度: [pixels, groups, 2(Sig/Dark), 3(S/R1/R2)]
        hr_refl = calculate_refl(combined_hr, m_hr_s, hr_target_refl, 
                                 'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
        lr_refl = calculate_refl(combined_lr, m_lr_s, lr_target_refl, 
                                 'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
        # --- 3. 波段聚合 (Bands Aggregation) ---
        # 计算 10nm 和 25nm 带宽的 LR 数据
        # band_centers = np.arange(400, 905, 5)
        # bands_10nm = compute_bands(lr_refl[:, :, 0], ccalib['LR_WL'], band_centers, 10)
        # bands_25nm = compute_bands(lr_refl[:, :, 0], ccalib['LR_WL'], band_centers, 25)
        # --- 4. 辐射定标 (Radiance) ---
        # 计算公式: Rad = COEF * (Net_Counts / Integration_Time)
        hr_cal = calculate_rad(combined_hr, m_hr_s, ccalib['HR_COEF'], 
                               'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
        lr_cal = calculate_rad(combined_lr, m_lr_s, ccalib['LR_COEF'], 
                               'Integration_time', 'I1_Integration_time', 'I2_Integration_time')
        # --- 5. 保存 L1 级数据 (csv)---
        csv_name = os.path.splitext(os.path.basename(f))[0] + '_HR_REFL.csv'
        hr_refl_df = pd.DataFrame(hr_refl.reshape(hr_refl.shape[0], -1))
        hr_refl_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','REFL', csv_name), index=False,columns=None)
        csv_name_lr_refl = os.path.splitext(os.path.basename(f))[0] + '_LR_REFL.csv'
        lr_refl_df = pd.DataFrame(lr_refl.reshape(lr_refl.shape[0], -1))
        lr_refl_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','REFL', csv_name_lr_refl), index=False,columns=None)
        csv_name_hr_cal = os.path.splitext(os.path.basename(f))[0] + '_HR_CAL.csv'
        hr_cal_df = pd.DataFrame(hr_cal.reshape(hr_cal.shape[0], -1))
        hr_cal_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','CAL', csv_name_hr_cal), index=False,columns=None)
        csv_name_lr_cal = os.path.splitext(os.path.basename(f))[0] + '_LR_CAL.csv'
        lr_cal_df = pd.DataFrame(lr_cal.reshape(lr_cal.shape[0], -1))
        lr_cal_df.to_csv(os.path.join(path, 'PROCESSED', 'L1','CAL', csv_name_lr_cal), index=False,columns=None)