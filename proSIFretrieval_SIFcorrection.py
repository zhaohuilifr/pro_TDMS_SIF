# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.direction'] = 'in'

unit_sif = '(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)'
unit_sif_hemi = '(mW m$^{-2}$ nm$^{-1}$)'
unit_ja = '(μmol m$^{-2}$ s$^{-1}$)'
unit_par = '(μmol photon m$^{-2}$ s$^{-1}$)'
unit_gpp = '(μmol $CO_2$ m$^{-2}$ s$^{-1}$)'
unit_phiF = '(-)'
font_size_text = 22
font_size_text2 = 9
# color_b = '#76cdde'
# color_o = '#f18b37'
color_b = 'k'
color_o ='r'

def umol_to_mw(umol, wl):
    # Convert umol/m²/sr/nm to mW/m²/sr/nm
    # 1 umol/m²/sr/nm = 1e-6 mol/m²/sr/nm
    # Energy (J) = n (mol) * Avogadro's number * h * c / wl (m)
    # Power (W) = Energy (J) / time (s)
    # 1 W = 1 J/s
    # 1 mW = 1e-3 W
    # Therefore, mW/m²/sr/nm = umol/m²/sr/nm * 1e-6 * Avogadro's number * h * c / wl (m) * 1e3
    h = 6.62607015e-34  # Planck constant in J*s
    c = 2.99792458e8    # Speed of light in m/s
    avogadro = 6.02214076e23  # Avogadro's number in mol^-1
    wl_m = wl * 1e-9  # Convert nm to m
    mw = umol * 1e-6 * avogadro * h * c / wl_m * 1e3
    return mw

def get_density(df_full, idx, x_label, y_label, scatter_data, max_density, bins=30):
    doy = df_full.loc[idx, 'DOY_UTC'].to_numpy()
    doy_int = doy.astype(int)
    x = df_full.loc[idx, x_label].to_numpy()
    y = df_full.loc[idx, y_label].to_numpy()
    
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    color = df_full.loc[idx, 'frac_dif'].to_numpy()[valid]

    doy_int = doy_int[valid]
    rmse = np.sqrt(np.nanmean((x - y) ** 2))
    rrmse = rmse / np.nanmean(x) * 100
    # 计算doy中每一天的均值
    tmp = [doy_int, x, y]
    tmp = pd.DataFrame(tmp).T
    tmp.columns = ['doy', 'x', 'y']
    # 对于每一天， 计算有效点的数量，如果有效点数量小于5，则该天的均值为nan
    tmp_count = tmp.groupby('doy').size()
    tmp = tmp.merge(tmp_count.to_frame('count'), left_on='doy', right_index=True)
    tmp = tmp[tmp['count'] >= 3]

    tmp_mean = tmp.groupby('doy').mean().reset_index()
    tmp_mean = tmp_mean.dropna(subset=['x', 'y']).reset_index(drop=True)
    rmse_mean = np.sqrt(np.nanmean((tmp_mean['x'] - tmp_mean['y']) ** 2))
    rrmse_mean = rmse_mean / np.nanmean(tmp_mean['x']) * 100
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
    density = hist[x_bin, y_bin]
    # density = 0
    # get p_value using scipy.stats.pearsonr
    if len(x) > 3 and len(y) > 3:
        r, p_value = pearsonr(x, y)
    else:
        r, p_value = np.nan, np.nan
    stats1 = [r**2, p_value, len(x), rmse, rrmse]

    r, p_value = pearsonr(tmp_mean['x'], tmp_mean['y'])

    stats2 = [r**2, p_value, len(tmp_mean), rmse_mean, rrmse_mean]
    scatter_data.append((year, x, y, color, density, stats1 ,tmp_mean, stats2))
    max_density = max(max_density, np.nanmax(density))
    return scatter_data, max_density

def p_star(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

save_figs = r'E:\Articles_ZL\2_CASTANEA model\0_raw draft\Figures2.0'
datapath = r'E:\Datahub\Barbeau\Data_matched_new4'
# datapath = r'E:\Datahub\Barbeau\Data_matched_new4_gcc'
# datapath = r'E:\Datahub\Barbeau\Data_matched_new6'
# datapath = r'E:\Datahub\Barbeau\Data_matched_new6\Res_67_761_full_nodecalage_kDF6.5'

months = [6,7,8,9]
hour_start = 9
hour_end = 16
years = ['2022','2023', '2024', '2025'] # 

scatter_data, max_density = [], 0
scatter_data1, max_density1 = [], 0

for iyear, year in enumerate(years):
    exfile_full = os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_full_APAR.xlsx')
    # exfile_full = os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_LIF_APAR.xlsx')
    exfile_lif = os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_LIF_APAR.xlsx')
    # exfile_lif = os.path.join(datapath, f'Barbeau_{year}_matched_CASTANEA_full_APAR.xlsx')
    df_full = pd.read_excel(exfile_full, sheet_name='Sheet1')
    df_lif = pd.read_excel(exfile_lif, sheet_name='Sheet1')
    # get month from DOY
    df_full['DOY_UTC_int'] = df_full['DOY_UTC'].astype(int)
    df_full['month_UTC'] = df_full['DOY_UTC_int'].apply(lambda x: pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=x-1)).dt.month
    df_lif['DOY_UTC_int'] = df_lif['DOY_UTC'].astype(int)
    df_lif['month_UTC'] = df_lif['DOY_UTC_int'].apply(lambda x: pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=x-1)).dt.month

    ## GPP filter and smooth
    # apply 3-sigma filter to GPP for each day, if GPP is greater than mean + 3*std or less than mean - 3*std, set it to nan
    gpp = np.array(df_full['GPP'].to_numpy(), copy=True)
    doy_unique = np.unique(df_full['DOY_UTC_int'].to_numpy())
    for doy in doy_unique[1:]:
        idx = (df_full['DOY_UTC_int'] == doy) & (df_full['hour_UTC'] >= hour_start) & (df_full['hour_UTC'] <= hour_end)
        gpp_doy = gpp[idx].copy()
        gpp_doy[gpp_doy <=0] = np.nan # remove negative GPP values
        mean = np.nanmean(gpp_doy)
        std = np.nanstd(gpp_doy)
        idx_outlier = (gpp_doy > mean + 3 * std) | (gpp_doy < mean - 3 * std)
        gpp_doy[idx_outlier] = np.nan
        # moving window average with window size of 3
        gpp_doy = pd.Series(gpp_doy).rolling(window=3, min_periods=1, center=True).mean().to_numpy()
        gpp[idx]= gpp_doy
    df_full['GPP'] = gpp

    # # plot GPP and gpp 
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.scatter(df_full['DOY_UTC'], df_full['GPP'], c='blue', marker='.', s=10, alpha=0.5, label='GPP')
    # ax.scatter(df_full['DOY_UTC'],gpp, c='red', marker='o', s=20, alpha=0.5, label='GPP filtered')
    # plt.show()

    # df_full_corrected = df_full.copy()
    # tmp = df_full_corrected['']]

    # fAPAR top
    df_lif.loc[df_lif['Fs'] < 0.1, 'Fs'] = np.nan
    df_lif['fapar_top'] = df_lif['APARtotal_top'] / df_lif['PAR_total']
    df_lif['phi_fs'] = df_lif['Fs'] # / df_lif['fapar_top']
    

    # replace str value to nan
    df_full['SIFcanopy_760nm'] = pd.to_numeric(df_full['SIFcanopy_760nm'], errors='coerce')
    df_full['SIFtotal_760nm'] = pd.to_numeric(df_full['SIFtotal_760nm'], errors='coerce')
    # SIF units conversion
    # df_full['SIFcanopy_760nm'] = umol_to_mw(df_full['SIFcanopy_760nm']/300/np.pi, 760) * 0.2
    # df_full['SIFtotal_760nm'] = umol_to_mw(df_full['SIFtotal_760nm']/300/np.pi, 760) * 0.2
    df_full['SIFcanopy_760nm'] = umol_to_mw(df_full['SIFcanopy_760nm']/300/np.pi, 760)
    df_full['SIFtotal_760nm'] = umol_to_mw(df_full['SIFtotal_760nm']/300/np.pi, 760)
    # simulated APAR
    df_full['APAR_sim'] = df_full['PARdiraLAI'] + df_full['PARdifaLAI']
    # scale and clean SIF_3FLD
    # df_full.loc[df_full['SIF_3FLD'] > 0.8, 'SIF_3FLD'] = np.nan

    # filter data for DOY between 150 and 250 and hour between 8 and 17
    # idx_day = (df_full['DOY_UTC'] >=110) & (df_full['DOY_UTC'] <= 300)
    idx_day = (df_full['month_UTC'].isin(months))
    idx_hour = (df_full['hour_UTC'] >= hour_start) & (df_full['hour_UTC'] <= hour_end)
    idx_norain = df_full['Prec (mm)'] == 0
    idx_par = df_full['PAR (µmol/m2/s)'] >  200
    idx = idx_day & idx_hour & idx_norain & idx_par

    idx_day = (df_full['month_UTC'].isin(months))
    idx_hour = (df_full['hour_UTC'] >= hour_start) & (df_full['hour_UTC'] <= hour_end)
    idx_shade = (df_full['hour_UTC'] >= 10) & (df_full['hour_UTC'] < 12)
    idx_norain = df_full['Prec (mm)'] == 0
    idx_par = df_full['PAR (µmol/m2/s)'] >  0 # 200
    idx_sif = idx_day & idx_hour & idx_norain & idx_par & ( ~(idx_shade))

    # idx_day = (df_lif['DOY_UTC'] >=110) & (df_lif['DOY_UTC'] <= 300)
    idx_day = (df_lif['month_UTC'].isin(months))
    # idx_hour = (df_lif['hour_UTC'] >= 5) & (df_lif['hour_UTC'] <= 9)
    idx_hour = (df_lif['hour_UTC'] >= hour_start) & (df_lif['hour_UTC'] <= hour_end)
    idx_shade = (df_lif['hour_UTC'] >= 10) & (df_lif['hour_UTC'] < 12)
    idx_norain = df_lif['Prec (mm)'] == 0
    idx_par = df_lif['PAR (µmol/m2/s)'] >  0# 200
    idx1 = idx_day & idx_hour & idx_norain & idx_par & ( ~(idx_shade))

    idx_sky = df_lif['PAR_diffuse_fraction'] < 0.5  # Example threshold, adjust as needed
    idx2 = idx1  & idx_sky

    # get density for scatter plot
    scatter_data, max_density = get_density(df_full, idx_sif, 'PAR (µmol/m2/s)', 'SIF_3FLD', scatter_data, max_density, 30)
    scatter_data1, max_density1 = get_density(df_full, idx_sif, 'PAR (µmol/m2/s)', 'SIF_SFM_lin_a', scatter_data1, max_density1, 30)

max_density = max(max_density, max_density1)

# %% SIF only (four time periods, 8-10, 10-12, 12-14, 14-16)
fig, axs = plt.subplots(1, 4, figsize=(12, 5))
# fig.subplots_adjust(left=0.125, right=0.887, wspace=0.2, hspace=0.24, bottom = 0.15)
fig.subplots_adjust(left=0.125, right=0.95, wspace=0.1, hspace=0.24, bottom = 0.15)
for iyear, (year, x, y, color, density, stats1, tmp_mean, stats2) in enumerate(scatter_data):
    axs[iyear].scatter(x, y, c=density, s=20, cmap='viridis',
                       edgecolors='none', vmin=0, vmax=max_density)
    # axs[idata, iyear].scatter(x, y, c=color_b, s=20, edgecolors='none', alpha=0.5)
    color = np.where(color > 0.5, color_b, color_o)
    # axs[iyear].scatter(x, y, c=color, s=20, edgecolors='none', alpha=0.5, cmap='viridis', vmin=0, vmax=1)

    # axs[iyear].scatter(tmp_mean['x'], tmp_mean['y'], s=32, edgecolors=color_o, facecolors='none', alpha=0.8)
    # plot linear regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_reindex = np.linspace(np.nanmin(x)*0.9, np.nanmax(x)*1.1, 100)
    axs[iyear].plot(x_reindex, slope * x_reindex + intercept, color = color_b, linestyle='--', linewidth=2)
    # slope, intercept = np.polyfit(tmp_mean['x'], tmp_mean['y'], 1)
    # tmp_mean_reindex = np.linspace(np.nanmin(tmp_mean['x'])*0.9, np.nanmax(tmp_mean['x'])*1.1, 100)
    # axs[idata, iyear].plot(tmp_mean_reindex, slope * tmp_mean_reindex + intercept, color = color_o, linestyle='--', linewidth=2)

    # axs[iyear].set_title(f'{year}', fontsize=18, fontweight='bold')
    # axs[iyear].plot([0, 2], [0, 2], 'r:', linewidth=1)
    axs[iyear].set_xlim([0,2400])
    axs[iyear].set_ylim([0,2])

    # add text of r2, p_value, and number of valid points in the bottom right of each subplot
    # axs[iyear].text(0.45, 0.7, f'$N$ = {stats1[2]}\n$R^2$ = {stats1[0]:.2f}'+p_star(stats1[1]), transform=axs[iyear].transAxes, 
    #                        fontsize=font_size_text2, fontweight='bold', va='bottom', ha='left', color = color_b)
    # axs[iyear].text(0.45, 0.47, f'$N$ = {stats2[2]}\n$R^2$ = {stats2[0]:.2f}'+p_star(stats2[1]), transform=axs[iyear].transAxes, 
    #                        fontsize=font_size_text2, fontweight='bold', va='bottom', ha='left', color = color_o)
    
    if iyear > 0:
        axs[iyear].set_yticklabels([])

axs[0].set_ylabel('Measured SIF(SIF3) ' +'\n'+ unit_sif, fontsize=18)
fig.text(0.5, 0.05, 'PAR(fluxdata) ' + unit_par, ha='center', va='center', fontsize=18)
# axs[0].set_xlabel('PAR(fluxdata) ' + unit_par, fontsize=18)


# fig.savefig(save_figs + r'\fig5_SIF_scatter_density_color.pdf', dpi=600, bbox_inches='tight')
# fig.savefig(save_figs + r'\fig5_SIF_scatter_density_color.png', dpi=600, bbox_inches='tight')
# fig.savefig(save_figs + r'\fig5_SIF_scatter_density_color.svg', dpi=600, bbox_inches='tight')

plt.show()
