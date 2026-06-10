# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
import glob

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
# unit_rad = '\n'+'(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)'
# unit_ref = ' '+'(-)'
unit_rad = '\n'+'(µmol m$^{-2}$ nm$^{-1}$)'
unit_ref = ' '+'(-)'

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

def mW_to_umol(lamda, rad):
    # Convert radiance from mW/m2/nm/sr to µmol/m2/nm/sr
    # lamda: wavelength in nm
    # rad: radiance in mW/m2/nm/sr
    h = 6.62607015e-34 # Planck constant in J*s
    c = 299792458 # speed of light in m/s
    NA = 6.02214076e23 # Avogadro's number in mol^-1
    # Convert wavelength from nm to m
    lamda_m = np.asarray(lamda) * 1e-9
    # Calculate energy per photon in J
    E_photon = h * c / lamda_m
    # Convert radiance from mW/m2/nm/sr to W/m2/nm/sr
    rad_W = np.asarray(rad) * 1e-3
    # Calculate photon flux in µmol/m2/nm/sr
    if rad_W.ndim == 2 and rad_W.shape[1] == len(lamda):
        flux = rad_W / E_photon[np.newaxis, :]
    elif rad_W.ndim == 2 and rad_W.shape[0] == len(lamda):
        flux = rad_W / E_photon[:, np.newaxis]
    else:
        flux = rad_W / E_photon

    flux = flux * (1e6 / NA)  # photons -> µmol

    if isinstance(rad, pd.DataFrame):
        return pd.DataFrame(flux, index=rad.index, columns=rad.columns)
    return flux

def integrate_spectra(wavelengths, values, wl_min=400, wl_max=700):
    # Integrate the spectra over the specified wavelength range using the trapezoidal rule
    idx = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if len(values.shape) == 1:
        integrated_value = np.trapz(values[idx], wavelengths[idx])
    else:
        integrated_value = []
        for i in values.columns:
            integrated_value.append(np.trapz(values.loc[idx, i], wavelengths[idx]))
    return integrated_value 


def doy_to_datetime(year, doy):
    return datetime(year, 1, 1) + pd.to_timedelta(doy - 1, unit='D')

import numpy as np
print(np.__file__)
print('has trapz:', hasattr(np, 'trapz'))
if not hasattr(np, 'trapz'):
    def trapz(y, x):
        y = np.asarray(y)
        x = np.asarray(x)
        return np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0)
    np.trapz = trapz

years = ['2022', '2023', '2024', '2025']
months = [6, 7, 8, 9] # choose two days for each month, at noon time
sunny_days = [[167, 176], [189, 191], [223, 233], [249, 267]]
cloudy_days = [[171, 173], [186, 194], [213, 230], [254, 266]]
days = [sunny_days, cloudy_days]
days_label = ['sunny', 'cloudy']
dayhour_range = (8, 17) # 

year = '2022'
path_LR_REF = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\{year}\PROCESSED\L1\REFL'.format(year=year)
path_LR_RAD = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\{year}\PROCESSED\L1\CAL'.format(year=year)
path_MEAT = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\{year}\PROCESSED\L1\META'.format(year=year)
path_CAS = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis_new2\Res_95_761_{year}\SIF_canopy'.format(year=year)
savepath_figs = r'E:\Datahub\Barbeau\Data_matched_new2\figs\validation_spectra'
os.makedirs(savepath_figs, exist_ok=True)


# %% diurnal variation of spectra for all days

## ----------------------------------------------------------------
## check different ways of reflectance coefficient
## ----------------------------------------------------------------
day = 224 #233 # 213, 223 # do the whole season, 
datestr = doy_to_datetime(int(year), day).strftime('%Y%m%d')
lr_file_ref = glob.glob(os.path.join(path_LR_REF, datestr + '_LR_REFL.csv'))
lr_file_rad = glob.glob(os.path.join(path_LR_RAD, datestr + '_LR_CAL.csv'))
lr_meta = glob.glob(os.path.join(path_MEAT, datestr + '_LR_meta.csv'))
cas_file = glob.glob(os.path.join(path_CAS, "BAR_2022_2025_RU390_V5_2_no_stress_ok_SIF_"+year+"_"+str(day)+"*_canopy.csv"))
cas_file_lys = glob.glob(os.path.join(path_CAS.replace('SIF_canopy', 'SIF_layers'), "BAR_2022_2025_RU390_V5_2_no_stress_ok_SIF_"+year+"_"+str(day)+"*_lys_fo.csv"))
wl_lr = pd.read_csv(r'E:\Datahub\Barbeau\Data_SIF\Califiles\LR_WL.csv')['WL']
# reindex cas_file
tmp, tmp1 = [], []
for i, f in enumerate(cas_file):
    hour_cas = int(f.split('_')[-4]) / 100
    tmp.append((f, hour_cas))
    tmp1.append((cas_file_lys[i], hour_cas))
tmp = sorted(tmp, key=lambda x: x[1])
cas_file = [x[0] for x in tmp]
tmp1 = sorted(tmp1, key=lambda x: x[1])
cas_file_lys = [x[0] for x in tmp1]


df_lr_ref = pd.read_csv(lr_file_ref[0])
df_lr_rad = pd.read_csv(lr_file_rad[0])
df_meta = pd.read_csv(lr_meta[0])
# df_cas = pd.read_csv(cas_file[0])
df_meta['Time_mid'] = df_meta['Time_start'] + (df_meta['Time_end'] - df_meta['Time_start']) / 2
df_meta['DOY_vis'] = df_meta['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
df_meta['Hour'] = df_meta['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])
# idx = (df_meta['Hour'] >= 8) & (df_meta['Hour'] < 17)
hour = df_meta['Hour']

idx_rad = np.array(range(0, df_lr_rad.shape[1], 3)).astype('str')
idx_ird_bf = np.array(range(1, df_lr_rad.shape[1], 3)).astype('str')
idx_ird_af = np.array(range(2, df_lr_rad.shape[1], 3)).astype('str')
idx_ref_bf = np.array(range(0, df_lr_ref.shape[1], 2)).astype('str')
idx_ref_af = np.array(range(1, df_lr_ref.shape[1], 2)).astype('str')
spec_ref_bf = df_lr_ref.loc[:, idx_ref_bf]
spec_ref_bf.columns = range(spec_ref_bf.shape[1])
spec_ref_af = df_lr_ref.loc[:, idx_ref_af]
spec_ref_af.columns = range(spec_ref_af.shape[1])
spec_ird_bf = df_lr_rad.loc[:, idx_ird_bf]
spec_ird_bf.columns = range(spec_ird_bf.shape[1])
spec_ird_af = df_lr_rad.loc[:, idx_ird_af]
spec_ird_af.columns = range(spec_ird_af.shape[1])


spec_ref_noon = (spec_ref_bf + spec_ref_af) / 2 # average of before and after, to reduce noise
spec_rad_noon = (df_lr_rad.loc[:, idx_rad]) * 1e3  * np.pi # convert from W/m2/sr/nm to mW/m2/nm
spec_rad_noon.columns = range(spec_rad_noon.shape[1])
spec_ird_noon = (spec_ird_bf + spec_ird_af) / 2  * 1e3  * np.pi # convert from W/m2/sr/nm to mW/m2/nm
spec_ird_noon1 = spec_rad_noon/spec_ref_noon # unit : mW/m2/sr/nm

# wl_lr = df_lr_ref.columns.astype(float)
spec_rad_noon_umol = mW_to_umol(wl_lr, spec_rad_noon) # convert from mW/m2/sr/nm to µmol/m2/sr/nm
spec_ird_noon_umol = mW_to_umol(wl_lr, spec_ird_noon) # convert from mW/m2/sr/nm to µmol/m2/sr/nm

PAR_ird = integrate_spectra(wl_lr, spec_ird_noon_umol, wl_min=400, wl_max=700)
PARout_rad = integrate_spectra(wl_lr, spec_rad_noon_umol, wl_min=400, wl_max=700)


fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharey='row')
# Normalize hour values and create colormap
norm = mcolors.Normalize(vmin=hour.min(), vmax=hour.max())
cmap = cm.get_cmap('jet')

for i in range(spec_ref_noon.shape[1]):
    color = cmap(norm(hour.iloc[i]))
    idx = (wl_lr >= 400) & (wl_lr <= 900)
    if hour.iloc[i] < 8 or hour.iloc[i] >= 17:
        continue
    axs[0,0].plot(wl_lr[idx], spec_ref_noon.loc[idx, i], c=color, alpha = 0.3)
    axs[1,0].plot(wl_lr[idx], spec_rad_noon_umol.loc[idx, i], c=color, alpha = 0.3)
    axs[2,0].plot(wl_lr[idx], spec_ird_noon_umol.loc[idx, i], c=color, alpha = 0.3)
    # axs[1,0].plot(wl_lr, spec_rad_noon.iloc[i, :], c=color, alpha = 0.3)
    # axs[2,0].plot(wl_lr, spec_ird_noon.iloc[i, :], c=color, alpha = 0.3)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=axs, label='Hour of Day')

hour_cass = []
par_cas, par_cas_sim, parout_cas_sim = [], [], []
spec_rad_cas = []
spec_ird_cas = []
spec_ref_cas = []
j = 0
for i in range(0, len(cas_file)-1):
    cas = cas_file[i]
    df_cas = pd.read_csv(cas)
    df_cas['radiance_f'] = df_cas['radiance'] + df_cas['SIFcanopy']
    df_cas['reflectance_f'] = df_cas['radiance_f'] / df_cas['irradiance']
    df_cas_par = pd.read_csv(cas_file_lys[i + 1], index_col=False)   ################## need to check the time of lys and cas files, make sure they match
    
    print(f"Processing CAS file: {os.path.basename(cas)}")
    print(f"CAS PAR data: {os.path.basename(cas_file_lys[i + 1])}")
    hour_cas = int(cas.split('_')[-4]) / 100 - 1 # CET to UTC
    if hour_cas < 8 or hour_cas >= 17:
        continue
    par_cas.append([hour_cas, (df_cas_par['PARdiroLAI'].values[0] + df_cas_par['PARdifoLAI'].values[0])/0.2]) # convert from per LAI to per m2 ground, assuming LAI = 0.2
    # print(f"Hour: {hour_cas}" + df_cas_par['PARdiroLAI'].values[0], df_cas_par['PARdifoLAI'].values[0])
    hour_cass.append(hour_cas)
    spec_rad_cas.append([hour_cas, df_cas['radiance_f'].astype(float)]) # unit: umol/m2/nm
    spec_ird_cas.append([hour_cas, df_cas['irradiance'].astype(float)]) # unit: umol/m2/nm
    spec_ref_cas.append([hour_cas, df_cas['reflectance_f'].astype(float)/np.pi]) # hemispherical reflectance to directional reflectance, assuming Lambertian surface
    
    wl_cas = df_cas['wl']
    par_cas_sim.append([hour_cas, integrate_spectra(wl_cas, df_cas['irradiance'].values, wl_min=400, wl_max=700)])
    parout_cas_sim.append([hour_cas, integrate_spectra(wl_cas, df_cas['radiance_f'].values, wl_min=400, wl_max=700)])

hour_cass = sorted(hour_cass)
par_cas = sorted(par_cas, key=lambda x: x[0]) # sort by hour
par_cas_sim = sorted(par_cas_sim, key=lambda x: x[0]) # sort by hour
parout_cas_sim = sorted(parout_cas_sim, key=lambda x: x[0]) # sort by hour  
spec_rad_cas = sorted(spec_rad_cas, key=lambda x: x[0]) # sort by hour
spec_ird_cas = sorted(spec_ird_cas, key=lambda x: x[0]) # sort by hour
spec_ref_cas = sorted(spec_ref_cas, key=lambda x: x[0]) # sort by hour
for i in range(len(spec_rad_cas)):
    color = cmap(norm(hour_cass[i]))
    axs[0,1].plot(wl_cas.T, spec_ref_cas[i][1], label='Reflectance(sim)', c=color)
    axs[1,1].plot(wl_cas.T, spec_rad_cas[i][1], label='Radiance(sim)', c=color)
    axs[2,1].plot(wl_cas.T, spec_ird_cas[i][1], label='Irradiance(sim)', c=color)



for i in range(2):
    axs[0, i].set_title(f'{datestr}, DOY {day}, {days_label[0]}')
    for j in range(3):
        # axs[j, 0].legend()
        axs[j,i].set_xlim([395, 905])
axs[0, 0].set_ylabel('Directional Reflectance' + unit_ref)
axs[1, 0].set_ylabel('Reflected Radiance'+unit_rad)
axs[2, 0].set_ylabel('Incident Radiance'+unit_rad)
axs[2, 0].set_xlabel('Wavelength (nm)')
axs[2, 1].set_xlabel('Wavelength (nm)')
# fig.savefig(os.path.join(savepath_figs, f'validation_spectra_diurnal_{year}_{day}.png'), dpi=300, bbox_inches='tight')

# %% SWout, PPFDin
path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel\data_gpp_with_uncertainties'
df_flux = pd.read_excel(os.path.join(path_flux, 'Barbeau_2022_LI7500_noAoA_uthvar_hh_DoubleInstrumentGF.xls'), sheet_name='data')
idx_flux = (df_flux['jj'] == day) & (df_flux['hh'] >= 9) & (df_flux['hh'] < 18)

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharey='row')
axs[0].plot(df_flux.loc[idx_flux, 'hh']-1, df_flux.loc[idx_flux, 'SWout (W/m²)']*2.1, label='SWout', c='k', marker = '+')
axs[0].plot(*zip(*parout_cas_sim), label='PARout (Integrated from CASTANEA spectra)', c='g', marker = 's')
axs[0].plot(hour, PARout_rad, label='PARout (Integrated from LR spectra)', c='b', marker = '.')
axs[0].set_xlabel('Hour of Day')
axs[0].set_ylabel('Reflected PAR (µmol/$m^2$/s)')
axs[0].legend()
# axs[1].plot(*zip(*par_cas), label='PAR(PQS1)', c='cyan', marker = 'o')
axs[1].plot(df_flux.loc[idx_flux, 'hh']-1, df_flux.loc[idx_flux, 'SWin (W/m²)']*2.1, label='SWin', c='k', marker = '+')
axs[1].plot(df_flux.loc[idx_flux, 'hh']-1, df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], label='PAR (PQS1)', c='r', marker = 'o')
axs[1].plot(*zip(*par_cas_sim), label='PAR (Integrated from CASTANEA spectra)', c='g', marker = 's')
axs[1].plot(hour, PAR_ird, label='PAR (Integrated from LR spectra)', c='b', marker = '.')
axs[1].set_xlabel('Hour of Day')
axs[1].set_ylabel('Incident PAR (µmol/$m^2$/s)')
axs[1].legend()
plt.show()

# %% 
# for month in months:
#     for days_i in days:
#         fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey='row')
#         for i, day in enumerate(days_i[months.index(month)]):
#             print(f"Processing {year} month {month} day {day}")
#             datestr = doy_to_datetime(int(year), day).strftime('%Y%m%d')
#             # read LR data
#             lr_file_ref = glob.glob(os.path.join(path_LR_REF, datestr + '_bandsR10nm.csv'))
#             lr_file_rad = glob.glob(os.path.join(path_LR_RAD, datestr + '_bandsRAD10nm.csv'))
#             lr_meta = glob.glob(os.path.join(path_MEAT, datestr + '_LR_meta.csv'))
#             # choose 11:00 hour for CAS data
#             cas_file = glob.glob(os.path.join(path_CAS, "BAR_2022_2025_RU390_V5_2_no_stress_ok_SIF_"+year+"_"+str(day)+"_1100*_canopy.csv"))

#             if lr_file_ref and lr_file_rad and lr_meta and cas_file:
#                 df_lr_ref = pd.read_csv(lr_file_ref[0])
#                 df_lr_rad = pd.read_csv(lr_file_rad[0]) 
#                 df_meta = pd.read_csv(lr_meta[0])
#                 df_cas = pd.read_csv(cas_file[0])
#                 df_meta['Time_mid'] = df_meta['Time_start'] + (df_meta['Time_end'] - df_meta['Time_start']) / 2
#                 df_meta['DOY_vis'] = df_meta['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
#                 df_meta['Hour'] = df_meta['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])

#                 idx = (df_meta['Hour'] >= 11.5) & (df_meta['Hour'] < 12.5)
#                 spec_ref_noon = np.mean(df_lr_ref[idx], axis = 0) 
#                 spec_rad_noon = np.mean(df_lr_rad[idx], axis = 0) * 1e3 # convert from W/m2/sr/nm to mW/m2/sr/nm
#                 spec_ird_noon = spec_rad_noon/spec_ref_noon
#                 wl_lr = df_lr_ref.columns.astype(float)  - 10 # ?
#                 spec_rad_noon_umol = mW_to_umol(wl_lr, spec_rad_noon) # convert from mW/m2/sr/nm to µmol/m2/sr/nm
#                 spec_ird_noon_umol = mW_to_umol(wl_lr, spec_ird_noon) # convert from mW/m2/sr/nm to µmol/m2/sr/nm

#                 spec_rad_cas = df_cas['radiance'].astype(float) 
#                 spec_ird_cas = df_cas['irradiance'].astype(float) 
#                 spec_ref_cas = df_cas['reflectance'].astype(float)
#                 wl_cas = df_cas['wl']

#                 axs[0, i].plot(wl_lr, spec_ref_noon, label='Reflectance(mea)')
#                 axs[0, i].plot(wl_cas, spec_ref_cas, label='Reflectance(sim)')
#                 axs[1, i].plot(wl_lr, spec_rad_noon, label='Reflected Radiance(mea)')
#                 axs[1, i].plot(wl_cas, spec_rad_cas, label='Reflected Radiance(sim)')
#                 axs[2, i].plot(wl_lr, spec_ird_noon, label='Incident Radiance(mea)')
#                 axs[2, i].plot(wl_cas, spec_ird_cas, label='Incident Radiance(sim)')

#         for i in range(2):
#             axs[0, i].set_title(f'{datestr}, DOY {days_i[months.index(month)][i]}, {days_label[days.index(days_i)]}')
#             for j in range(3):
#                 axs[j, 0].legend()
#                 axs[j,i].set_xlim([395, 905])
#         axs[0, 0].set_ylabel('Reflectance' + unit_ref)
#         axs[1, 0].set_ylabel('Radiance'+unit_rad)
#         axs[2, 0].set_ylabel('Irradiance'+unit_rad)
#         axs[2, 0].set_xlabel('Wavelength (nm)')
#         axs[2, 1].set_xlabel('Wavelength (nm)')
        
#         fig.savefig(os.path.join(savepath_figs, f'validation_spectra_{year}_{month}_{days_label[days.index(days_i)]}.png'), dpi=300, bbox_inches='tight')
#     # plt.show()
#     # print('a')