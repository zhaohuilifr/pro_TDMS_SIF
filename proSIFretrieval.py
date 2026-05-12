# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import datetime
import os
import glob
import numpy as np
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor
sys.path.append(r'E:\Codehub\functions')
import SIFretrieval, VIs_calculation_func

unit_rad = '$mW/m^2/sr/nm$'
unit_irrd = '$mW/m^2/nm$'
unit_sif = '$mW/m^2/sr/nm$'
unit_vis = '-' 


def build_file_map(file_list, marker):
    """Build a fast basename-prefix -> filepath lookup map."""
    file_map = {}
    for fpath in file_list:
        prefix = os.path.basename(fpath).split(marker)[0]
        file_map[prefix] = fpath
    return file_map

def process_hr_sif_day(task):
    """Process one HR daily SIF file and write the daily result CSV."""
    hr_cal_csv, hr_meta_csv, hr_refl_csv, wl_hr_values, savepath = task

    hr_cal = pd.read_csv(hr_cal_csv)
    hr_meta = pd.read_csv(hr_meta_csv)
    hr_refl = pd.read_csv(hr_refl_csv)

    # time_start_dt = matlab2datetime(hr_meta['Time_start'])
    # time_end_dt = matlab2datetime(hr_meta['Time_end'])
    time_start_dt = hr_meta['Time_start']
    time_end_dt = hr_meta['Time_end']
    time_mid_dt = time_start_dt + (time_end_dt - time_start_dt) / 2

    radiance_arr = hr_cal.iloc[:, 0::3].to_numpy(dtype=float) * 1e3 # convert from $W/m^2/nm/sr$ to $mW/m^2/nm/sr$
    irradiance_before_arr = hr_cal.iloc[:, 1::3].to_numpy(dtype=float) * 1e3 * np.pi  # convert from $W/m^2/nm$ to $mW/m^2/nm/sr$
    irradiance_after_arr = hr_cal.iloc[:, 2::3].to_numpy(dtype=float) * 1e3 * np.pi  # convert from $W/m^2/nm$ to $mW/m^2/nm/sr$
    reflectance_before_arr = hr_refl.iloc[:, 0::2].to_numpy(dtype=float)
    reflectance_after_arr = hr_refl.iloc[:, 1::2].to_numpy(dtype=float)

    irrad_arr = np.nanmean(
        np.stack((irradiance_before_arr, irradiance_after_arr), axis=0),
        axis=0
    )
    ref_arr = np.nanmean(
        np.stack((reflectance_before_arr, reflectance_after_arr), axis=0),
        axis=0
    )

    # SIFresults_daily = hr_meta[meta_cols].copy()
    SIFresults_daily = hr_meta.copy() # keep all metadata columns for now
    n_steps = radiance_arr.shape[1]
    sif_3fld = np.full(n_steps, np.nan)
    sif_ifld = np.full(n_steps, np.nan)
    sif_sfm_lin_a = np.full(n_steps, np.nan)
    sif_sfm_qua_a = np.full(n_steps, np.nan)
    sif_sfm_lin_b = np.full(n_steps, np.nan)
    sif_sfm_qua_b = np.full(n_steps, np.nan)

    for idx in range(n_steps):
        SIFcon = SIFretrieval.container()
        SIFcon.wl = wl_hr_values
        # SIFcon.ftime = time_mid_dt.iloc[idx]
        SIFcon.irrad = irrad_arr[:, idx]
        SIFcon.rad = radiance_arr[:, idx]
        SIFcon.ref = ref_arr[:, idx]

        if np.isnan(SIFcon.irrad).all() or np.isnan(SIFcon.rad).all() or np.isnan(SIFcon.ref).all():
            continue

        fs_sfm = SIFretrieval.SFM(SIFcon, wlsfm)
        sif_3fld[idx] = SIFretrieval.FLD3(SIFcon, wl3fld, widths)
        sif_ifld[idx] = SIFretrieval.iFLD(SIFcon, wlin, fwhm)
        sif_sfm_lin_a[idx] = fs_sfm[0]
        sif_sfm_qua_a[idx] = fs_sfm[1]
        sif_sfm_lin_b[idx] = fs_sfm[2]
        sif_sfm_qua_b[idx] = fs_sfm[3]

    SIFresults_daily['Time_mid'] = time_mid_dt.values
    SIFresults_daily['SIF_3FLD'] = sif_3fld
    SIFresults_daily['SIF_iFLD'] = sif_ifld
    SIFresults_daily['SIF_SFM_lin_a'] = sif_sfm_lin_a
    SIFresults_daily['SIF_SFM_qua_a'] = sif_sfm_qua_a
    SIFresults_daily['SIF_SFM_lin_b'] = sif_sfm_lin_b
    SIFresults_daily['SIF_SFM_qua_b'] = sif_sfm_qua_b

    cols = SIFresults_daily.columns.tolist()
    cols.insert(cols.index('Time_end') + 1, cols.pop(cols.index('Time_mid')))
    SIFresults_daily = SIFresults_daily[cols]
    SIFresults_daily['Time_start'] = time_start_dt
    SIFresults_daily['Time_end'] = time_end_dt

    SIFresults_daily.loc[SIFresults_daily['SIF_3FLD'] < 0, 'SIF_3FLD'] = np.nan
    SIFresults_daily.loc[SIFresults_daily['SIF_iFLD'] < 0, 'SIF_iFLD'] = np.nan
    SIFresults_daily.loc[SIFresults_daily['SIF_SFM_lin_a'] < 0, 'SIF_SFM_lin_a'] = np.nan
    SIFresults_daily.loc[SIFresults_daily['SIF_SFM_qua_a'] < 0, 'SIF_SFM_qua_a'] = np.nan
    SIFresults_daily.loc[SIFresults_daily['SIF_SFM_lin_b'] < 0, 'SIF_SFM_lin_b'] = np.nan
    SIFresults_daily.loc[SIFresults_daily['SIF_SFM_qua_b'] < 0, 'SIF_SFM_qua_b'] = np.nan

    daily_savepath = os.path.join(
        savepath,
        'Daily',
        os.path.basename(hr_cal_csv).replace('_CAL', '_SIF_Daily')
    )
    SIFresults_daily.to_csv(daily_savepath, index=False)
    return daily_savepath

def process_lr_vi_day(task):
    """Process one LR daily VI file and write the daily result CSV."""
    lr_cal_csv, lr_meta_csv, lr_refl_csv, wl_lr_values, savepath = task
    # import cal, meta and refl data for each day
    lr_cal = pd.read_csv(lr_cal_csv)
    lr_meta = pd.read_csv(lr_meta_csv)
    lr_refl = pd.read_csv(lr_refl_csv)
    # get time information
    # time_start = lr_meta['Time_start']
    # time_end = lr_meta['Time_end']
    # time_start_dt = matlab2datetime(time_start)
    # time_end_dt = matlab2datetime(time_end)
    time_start_dt = lr_meta['Time_start']
    time_end_dt = lr_meta['Time_end']
    time_mid_dt = time_start_dt + (time_end_dt - time_start_dt) / 2
    # %% get radiance, irradiance and reflectance
    radiance = lr_cal.iloc[:, [i for i in range(0, lr_cal.shape[1], 3)]] * 1e3 # convert from $W/m^2/nm/sr$ to $mW/m^2/nm/sr$
    radiance.columns = range(radiance.shape[1])
    # irradiance_before = lr_cal.iloc[:, [i+1 for i in range(0, lr_cal.shape[1], 3)]] * 1e3 * np.pi # convert from $W/m^2/nm$ to $mW/m^2/nm/sr$
    # irradiance_before.columns = range(irradiance_before.shape[1])
    # irradiance_after = lr_cal.iloc[:, [i+2 for i in range(0, lr_cal.shape[1], 3)]] * 1e3 * np.pi # convert from $W/m^2/nm$ to $mW/m^2/nm/sr$
    # irradiance_after.columns = range(irradiance_after.shape[1])
    reflectance_before = lr_refl.iloc[:, [i for i in range(0, lr_refl.shape[1], 2)]]
    reflectance_before.columns = range(reflectance_before.shape[1])
    reflectance_after = lr_refl.iloc[:, [i+1 for i in range(0, lr_refl.shape[1], 2)]]
    reflectance_after.columns = range(reflectance_after.shape[1])

    # %% VIs calculation
    # VIsresults_daily = lr_meta[meta_cols_vis].copy()
    VIsresults_daily = lr_meta.copy() # keep all metadata columns for now
    # new empty container for VIs calculation
    VIcon = VIs_calculation_func.VI_container()
    # fill the container with data for each time step
    # wl = wl_lr['WL'].values
    # VIcon.irrad = np.nanmean([irradiance_before.iloc[:, idx].values, 
    #                           irradiance_after.iloc[:, idx].values], axis=0) * 1e3 * np.pi # convert from $W/m^2/nm/sr$ to $mW/m^2/nm$
    # rad = radiance * 1e3 #  convert from $W/m^2/nm/sr$ to $mW/m^2/nm/sr$
    ref = pd.DataFrame(np.nanmean([reflectance_before.values, reflectance_after.values], axis=0), 
                    columns=reflectance_before.columns)
    # perform VIs calculation using different methods
    VIcon = VIs_calculation_func.get_vegetation_indices2D(VIcon, wl_lr_values, ref, radiance)
    # store results in the daily results dataframe
    VIsresults_daily['Time_mid'] = time_mid_dt
    VIsresults_daily['NDVI'] = VIcon.NDVI
    VIsresults_daily['EVI'] = VIcon.EVI
    VIsresults_daily['MTCI'] = VIcon.MTCI
    VIsresults_daily['MTVI2'] = VIcon.MTVI2
    VIsresults_daily['PRI'] = VIcon.PRI
    VIsresults_daily['greenNDVI'] = VIcon.greenNDVI
    VIsresults_daily['rededgeNDVI'] = VIcon.rededgeNDVI
    VIsresults_daily['OSAVI'] = VIcon.OSAVI
    VIsresults_daily['CIgreen'] = VIcon.CIgreen
    VIsresults_daily['CIrededge'] = VIcon.CIrededge
    VIsresults_daily['CVI'] = VIcon.CVI
    VIsresults_daily['SR'] = VIcon.SR
    VIsresults_daily['NIRv'] = VIcon.NIRv
    VIsresults_daily['FCVI'] = VIcon.FCVI
    VIsresults_daily['NIRVR'] = VIcon.NIRVR
    ## save daily results to csv
    # move Time_mid column after Time_end
    cols = VIsresults_daily.columns.tolist()
    cols.insert(cols.index('Time_end') + 1, cols.pop(cols.index('Time_mid')))
    VIsresults_daily = VIsresults_daily[cols]
    # replace matlab time with python datetime in the results dataframe
    # VIsresults_daily['Time_start'] = time_start_dt
    # VIsresults_daily['Time_end'] = time_end_dt
    # save daily results to csv
    daily_savepath = os.path.join(savepath, 'Daily', os.path.basename(lr_cal_csv).replace('_CAL', '_VIs_Daily'))
    VIsresults_daily.to_csv(daily_savepath, index=False)

    return daily_savepath

# function to matlab datetime to python datetime
def matlab2datetime(matlab_datenum):
    # e.g. 739449.88170052 to 2024-07-17T05:09:40
    python_datetime = pd.to_datetime(matlab_datenum - 719529, unit='D')
    return python_datetime

# columns extracted from metadata
meta_cols = ['idx', 'Time_start', 'Time_end', 'PAR_micromoles_m_2s_1', 'Sunflag',
             'IRT0_SurfaceTemperature_celsius','IRT0_SensorTemperature_celsius', 
             'BOX_Temperature_celsius','BOX_RelativeHumidity_percent', 
             'HR_peltierCurrent_ampere', 'HR_radiatorTemperature_celsius', 
             'sun_var', 'nbdark_sig','nbsp_bad','dark_dpix_mean', 
             'dark_dpix_std', 'sp_dpix_mean', 'sp_dpix_std', 'prcdarkpix_sig', 
             'prcsppix_nosig', 'prcsppix_sat']
meta_cols_vis = ['idx', 'Time_start', 'Time_end',  
             'sun_var', 'nbdark_sig','nbsp_bad','dark_dpix_mean', 
             'dark_dpix_std', 'sp_dpix_mean', 'sp_dpix_std', 'prcdarkpix_sig', 
             'prcsppix_nosig', 'prcsppix_sat']

## parameters for SIF retrieval
# 3FLD parameters 
wl3fld = [752, 762, 775]
widths = [3.5, 2, 5]
# iFLD parameters
wlin,fwhm = 762, 0.15 # 762, 0.132
# SFM parameters
wlout1a,wlina,wlout2a = 755, 762, 768
wlout1b,wlinb,wlout2b = 685, 687, 690
wlsfm = [wlout1a,wlina,wlout2a, wlout1b,wlinb,wlout2b]
# Specfit parameters

# SVD parameters

# BSF parameters

if __name__ == "__main__":
    # %% ---------------------------- 数据文件路径设置 ------------------ ------------------ #  
    # Year = 2024 # 2022, 2023, 2024, 2025
    for Year in [2022]: # , 2023, 2024, 2025
        path = os.path.join(r'E:\Datahub\Barbeau\Data_SIF\SIF3data', str(Year),'PROCESSED\L1')
        savepath = os.path.join(r'E:\Datahub\Barbeau\Data_SIF\SIF3data', str(Year),'PROCESSED\L2')
        for level in ['Daily','Yearly']:
            os.makedirs(os.path.join(savepath, level), exist_ok=True)
        # 1. import metadata and calibration data
        folder_BANDS = os.path.join(path, 'BANDS')
        folder_META = os.path.join(path, 'META')
        folder_CAL = os.path.join(path, 'CAL')
        folder_REFL = os.path.join(path, 'REFL')
        # wavelength data
        wl_hr = pd.read_csv(os.path.join(folder_BANDS, 'HR_WL.csv'))
        wl_lr = pd.read_csv(os.path.join(folder_BANDS, 'LR_WL.csv'))
        # metadata
        hr_meta_csvs = glob.glob(os.path.join(folder_META, '*HR_meta.csv'))
        lr_meta_csvs = glob.glob(os.path.join(folder_META, '*LR_meta.csv'))
        # radiation data
        hr_cal_csvs = glob.glob(os.path.join(folder_CAL, '*HR_CAL.csv'))
        lr_cal_csvs = glob.glob(os.path.join(folder_CAL, '*LR_CAL.csv'))
        # reflectance data
        hr_refl_csvs = glob.glob(os.path.join(folder_REFL, '*HR_REFL.csv'))
        lr_refl_csvs = glob.glob(os.path.join(folder_REFL, '*LR_REFL.csv'))

        hr_meta_map = build_file_map(hr_meta_csvs, '_meta.csv')
        hr_refl_map = build_file_map(hr_refl_csvs, '_REFL.csv')
        lr_meta_map = build_file_map(lr_meta_csvs, '_meta.csv')
        lr_refl_map = build_file_map(lr_refl_csvs, '_REFL.csv')

        # %% ### SIF retrieval for each csv file 
        # Loop through each calibration csv file, perform SIF retrieval and save results to csv
        hr_tasks = []
        for hr_cal_csv in hr_cal_csvs:
            cal_prefix = os.path.basename(hr_cal_csv).split('_CAL')[0]
            hr_tasks.append((hr_cal_csv, hr_meta_map[cal_prefix], hr_refl_map[cal_prefix], wl_hr['WL'].to_numpy(), savepath))
        # os.cpu_count() = 32
        worker_count = max(1, (os.cpu_count() or 2) - 10) # leave some cores free
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            hr_daily_paths = list(executor.map(process_hr_sif_day, hr_tasks))

        # hr_daily_paths = list(glob.glob(os.path.join(savepath, 'Daily', '*HR_SIF_Daily.csv')))

        SIFresults_yearly = [pd.read_csv(path) for path in hr_daily_paths]
        SIFresults_yearly_df = pd.concat(SIFresults_yearly, ignore_index=True)
        SIFresults_yearly_df['idx'] = range(1, len(SIFresults_yearly_df) + 1)
        # save yearly results to csv
        yearly_savepath = os.path.join(savepath, 'Yearly', f'PROSIF_SIFresults_{Year}_Yearly.csv')
        SIFresults_yearly_df.to_csv(yearly_savepath, index=False)


        # %% ### VIs calculation for each csv file
        # Loop through each calibration csv file, perform VIs calculation and save results to csv
        lr_tasks = []
        for lr_cal_csv in lr_cal_csvs:
            cal_prefix = os.path.basename(lr_cal_csv).split('_CAL')[0]
            lr_tasks.append((lr_cal_csv, lr_meta_map[cal_prefix], lr_refl_map[cal_prefix], wl_lr['WL'].to_numpy(), savepath))
        
        worker_count = max(1, (os.cpu_count() or 2) - 10) # leave some cores free
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            lr_daily_paths = list(executor.map(process_lr_vi_day, lr_tasks))
        
        VIsresults_yearly = [pd.read_csv(path) for path in lr_daily_paths]
        VIsresults_yearly_df = pd.concat(VIsresults_yearly, ignore_index=True)
        VIsresults_yearly_df['idx'] = range(1, len(VIsresults_yearly_df) + 1)
        # save yearly results to csv
        yearly_savepath = os.path.join(savepath, 'Yearly', f'PROSIF_VIsresults_{Year}_Yearly.csv')
        VIsresults_yearly_df.to_csv(yearly_savepath, index=False)