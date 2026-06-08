# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

def func_ql_par(alpha_ql, hf_par, par):
    beta_ql = hf_par/np.log(2)
    ql = alpha_ql* np.exp(-1/beta_ql*par)
    return ql

def ql_par_func(par, alpha, beta):
    return alpha*np.exp(-beta*par)

def qL_model_exponential_linear(X, a, b):
    PAR, T = X
    # a - b * T 描述温度对闭合速率的负调节作用
    # k = a - b * T
    k = a - b * T
    return 0.2 + (0.85 - 0.2) * np.exp(-k * PAR)

def qL_model_exponential(X, a, b):
    PAR, T = X
    # a - b * T 描述温度对闭合速率的负调节作用
    # k = a - b * T
    k = a * np.exp(-b * T)
    return 0.2 + (0.85 - 0.2) * np.exp(-k * PAR)

def qL_model_exponential_minmax(X, qLmin, qLmax, a, b):
    PAR, T = X
    # a - b * T 描述温度对闭合速率的负调节作用
    # k = a - b * T
    k = a * np.exp(-b * T)
    return qLmin + (qLmax - qLmin) * np.exp(-k * PAR)

def qL_model_exponential_1(X,qLmin, qLmax, k):
    PAR, T = X
    return qLmin + (qLmax - qLmin) * np.exp(-k/T * PAR)

def qL_model_polynomial(X, a, b, c):
    PAR, T = X
    return a + b * PAR + c / PAR**2
    # return (a + b * PAR) /(1+ c / PAR**2)

# def qL_model_dynamic_ceiling(X, a, b, c, d):
#     PAR, T = X
#     # a - b * T 描述温度对闭合速率的负调节作用
#     qL_max = c + d * T
#     k = a * np.exp(-b * T)
#     return 0.2 + (qL_max - 0.2) * np.exp(-k * PAR)

def qL_model_dynamic_ceiling(X, qlmin, a, b, c, d):
    PAR, T = X
    # a - b * T 描述温度对闭合速率的负调节作用
    qL_max = c + d * T
    k = a * np.exp(-b * T)
    return qlmin + (qL_max - qlmin) * np.exp(-k * PAR)

def fit_x_y_curve(x, y, func, p0, bounds):
    # if x or y is empty
    if x.dropna().empty or y.dropna().empty:
        return [np.nan]*len(p0), np.nan
    else:
        tmp = pd.concat([x, y], axis=1).dropna()
        popt, pcov = curve_fit(func, tmp.iloc[:, 0], tmp.iloc[:, 1], p0=p0, maxfev=10000, bounds=bounds)
        residuals = tmp.iloc[:, 1] - func(tmp.iloc[:, 0], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tmp.iloc[:, 1] - np.mean(tmp.iloc[:, 1]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / len(tmp))
        perr = np.sqrt(np.diag(pcov)) # 参数标准误差
        return popt, r_squared, rmse, perr

yearstr = '2022'
path_match = r'E:\Datahub\Barbeau\Data_matched_new2'
savepath_figs = r'E:\Datahub\Barbeau\Data_matched_new2\figs\ql_vs_PAR_{year}'.format(year=yearstr)
os.makedirs(savepath_figs, exist_ok=True)
df_matched = pd.read_excel(os.path.join(path_match, f'Barbeau_{yearstr}_matched_CASTANEA.xlsx'))

df_matched.loc[df_matched['Fs'] < 0.15, 'Fs'] = np.nan
df_matched.loc[df_matched['SIFPSIILAI_yield_top'] < 0 , 'SIFPSIILAI_yield_top'] = np.nan
df_matched.loc[df_matched['SIFPSIILAI_yield_top'] > 0.03 , 'SIFPSIILAI_yield_top'] = np.nan
sifpsii_max = df_matched['SIFPSIILAI_yield_top'].max()
sifpsii_min = df_matched['SIFPSIILAI_yield_top'].min()
# scale Fs to the same range as SIFPSIILAI_yield_top
df_matched['Fs_scaled'] = (df_matched['Fs'] - df_matched['Fs'].min()) \
    / (df_matched['Fs'].max() - df_matched['Fs'].min()) * (sifpsii_max - sifpsii_min) + sifpsii_min

sifyield = df_matched['Fs_scaled'].values
apar = df_matched['PARdiraLAI'] + df_matched['PARdifaLAI']
sifpsii = sifyield * apar
phiPSIImax = 0.81
kDF = 10
ql = df_matched['JaLAI'] * (1 - phiPSIImax) / (phiPSIImax * (1 + kDF) * sifpsii)

idx = (df_matched['hour_UTC'] >= 3) & (df_matched['hour_UTC'] <= 17) & (df_matched['PAR (µmol/m2/s)'] > 0)
idx_day = idx & (df_matched['DOY_UTC'] >= 160) & (df_matched['DOY_UTC'] <= 185) # 150， 250

# %% 把每天的qL归一化到0.2 - 0.85的范围内
ql_day = ql[idx_day]

# %% fit qL-PAR curve for the selected days
p0 = [0, 0.001]  # 用数据最大值作为alpha初始值
bounds = ([0, 0], [1.0, 0.01])     # 放宽上界
popt, r_squared, rmse, perr = fit_x_y_curve(apar[idx_day], ql_day, ql_par_func, p0, bounds)
alpha_ql, beta_ql, r_squared_qL = popt[0], popt[1], r_squared
ql_fit = ql_par_func(apar[idx_day], *popt)
hf_par_fit = 1/beta_ql * np.log(2)

par_range = np.linspace(0, 2000, 100)
ql_dir = ql_par_func(par_range, 0.89, 7.68e-4)
ql_dif = ql_par_func(par_range, 0.6, 5.68e-4)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
#            c = df_matched.loc[idx_day,'DOY_UTC'], cmap='viridis', s=20, edgecolor='none')
# ax.scatter(apar[idx_day], ql_fit, color='red', marker = '.', label=f'Fitted parameters: alpha={alpha_ql:.3f}, beta={beta_ql:.5f}, hf_par={hf_par_fit:.1f}, R²={r_squared_qL:.3f}')
# ax.plot(par_range, ql_dir, color='blue', linestyle='--', label='Typical qL-PAR curve (dir)')
# ax.plot(par_range, ql_dif, color='green', linestyle='--', label='Typical qL-PAR curve (dif)')

# cbar = plt.colorbar(ax.collections[0], ax=ax)
# cbar.set_label('Day of Year (UTC)')
# ax.set_xlabel('PAR (µmol/m2/s)')
# ax.set_ylabel('qL (unitless)')
# ax.legend()
# ax.set_title(f'qL vs PAR for DOY 150-250 in {yearstr}')
# # fig.savefig(os.path.join(savepath_figs, f'qL_vs_PAR_DOY_150_250_DOYcolor.jpg'), dpi=300)



# qL_min 限制在 [0, 0.5], a 限制在 [0, 1], b 限制在 [0, 1]
# initial_guess = [0.005, 0.0001]
# bounds = ([0, 0], [0.5, 10])
# idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 192) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_exponential_linear, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_exponential_linear((par_demo, t_demo), *popt)
# print(f'Fitted parameters: a={popt[0]:.7f}, b={popt[1]:.7f}')

# initial_guess = [0.01, 0.05]
# bounds = ([1e-5, 1e-5], [0.5, 10])
# idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 192) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_exponential, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_exponential((par_demo, t_demo), *popt)
# print(f'Fitted parameters: a={popt[0]:.7f}, b={popt[1]:.7f}')

# initial_guess = [0.1, 0.8, 0.01, 0.05]
# bounds = ([0, 0, 1e-5, 1e-5], [0.5, 100, 0.5, 10])
# idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 192) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_exponential_minmax, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_exponential_minmax((par_demo, t_demo), *popt)
# print(f'Fitted parameters: qLmin={popt[0]:.7f}, qLmax={popt[1]:.7f}, a={popt[2]:.7f}, b={popt[3]:.7f}')


# initial_guess = [ 0.05, 0.4, 0.025]
# bounds = (
#     [  1e-5, 0.1,  0.0],   # 下界 (限制d>=0，确保温度越高天花板越高)
#     [  0.5,  1.0,  0.08]   # 上界
# )
# # idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 198) # 150， 250
# idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] < 250) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_polynomial, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_polynomial((par_demo, t_demo), *popt)
# print(f'Fitted parameters: a={popt[0]:.7f}, b={popt[1]:.7f}, c={popt[2]:.7f}')


initial_guess = [0,0.01, 0.05, 0.4, 0.025]
bounds = (
    [ 0,1e-5, 1e-5, 0.1,  0.0],   # 下界 (限制d>=0，确保温度越高天花板越高)
    [ 1, 0.1,  0.5,  1.0,  0.08]   # 上界
)
# idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 198) # 150， 250
idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] < 200) # 150， 250
par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
qL_demo = ql[idx_day]
# remove nan values
idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
par_demo = par_demo[idx_valid]
t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
qL_demo = qL_demo[idx_valid]
popt, pcov = curve_fit(qL_model_dynamic_ceiling, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
qL_fitted = qL_model_dynamic_ceiling((par_demo, t_demo), *popt)

print(f'Fitted parameters: a={popt[0]:.7f}, b={popt[1]:.7f}, c={popt[2]:.7f}, d={popt[3]:.7f}')
residuals = qL_demo - qL_fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((qL_demo - np.mean(qL_demo))**2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(ss_res / len(qL_demo))
perr = np.sqrt(np.diag(pcov)) # 参数标准误差


# initial_guess = [0.1, 1.0, 0.01]
# bounds = (
#     [ 0, 0.5, 0,],   # 下界 (限制d>=0，确保温度越高天花板越高)
#     [ 0.5, 10,  1]   # 上界
# )
# # idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 198) # 150， 250
# idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] < 200) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_exponential_1, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_exponential_1((par_demo, t_demo), *popt)
# print(f'Fitted parameters: qLmin={popt[0]:.7f}, qLmax={popt[1]:.7f}, k={popt[2]:.7f}')

# initial_guess = [0.1, 0.01, 0.05, 0.4, 0.025]
# bounds = (
#     [ 0, 0, 0, 0.1,  0.0],   # 下界 (限制d>=0，确保温度越高天花板越高)
#     [ 0.5, 0.1,  0.5,  1.0,  0.08]   # 上界
# )
# # idx_day = idx & (df_matched['DOY_UTC'] >= 191) & (df_matched['DOY_UTC'] < 198) # 150， 250
# idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] < 200) # 150， 250
# par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
# qL_demo = ql[idx_day]
# # remove nan values
# idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
# par_demo = par_demo[idx_valid]
# t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
# qL_demo = qL_demo[idx_valid]
# popt, pcov = curve_fit(qL_model_dynamic_ceiling, (par_demo, t_demo), qL_demo, p0=initial_guess, bounds=bounds)
# qL_fitted = qL_model_dynamic_ceiling((par_demo, t_demo), *popt)
# print(f'Fitted parameters: qLmin ={popt[0]:.7f}, a={popt[1]:.7f}, b={popt[2]:.7f}, c={popt[3]:.7f}, d={popt[4]:.7f}')

# qL_fitted1 = qL_model_exponential((par_demo, t_demo), 0.0156, 5.3642) # 
# idx_day = idx & (df_matched['DOY_UTC'] >= 192) & (df_matched['DOY_UTC'] < 193) # 150， 250
# idx_day = idx & (df_matched['DOY_UTC'] >= 172) & (df_matched['DOY_UTC'] < 173) # 150， 250
# idx_day = idx & (df_matched['DOY_UTC'] >= 181) & (df_matched['DOY_UTC'] < 182) # 150， 250
idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] < 200) # 150， 250
par_demo = df_matched.loc[idx_day, 'PAR (µmol/m2/s)']
qL = df_matched['JaLAI'].copy()
qL_demo = ql[idx_day]
# remove nan values
idx_valid = ~np.isnan(par_demo) & ~np.isnan(qL_demo)
par_demo = par_demo[idx_valid]
t_demo = df_matched.loc[idx_day, 'Ta (°C)'][idx_valid]
qL_demo = qL_demo[idx_valid]
qL_fitted1 = qL_model_dynamic_ceiling((par_demo, t_demo), *popt)  
# qL_fitted1 = qL_model_exponential_1((par_demo, t_demo), *popt) 
# qL_fitted1 = qL_model_polynomial((par_demo, t_demo), *popt)

# get r2 and rmse
qL_r2 = 1 - np.sum((qL_demo - qL_fitted1) ** 2) / np.sum((qL_demo - np.mean(qL_demo)) ** 2)
qL_rmse = np.sqrt(np.mean((qL_demo - qL_fitted1) ** 2))
print(f'qL model R²: {qL_r2:.3f}, RMSE: {qL_rmse:.3f}')

fig, ax = plt.subplots(1,4, figsize=(20, 6))
ax[0].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
           c = df_matched.loc[idx_day,'Ta (°C)'], cmap='viridis', s=20, edgecolor='none')
# ax[0].scatter(par_demo, qL_fitted, s=20, edgecolor='k', facecolor='none', label = f'Fitted parameters: a={popt[0]:.3f}, b={popt[1]:.5f}')
# ax[0].scatter(par_demo, qL_fitted1, s=20, edgecolor='none', c= 'r',label='fitted' + f' R²={qL_r2:.3f}, RMSE={qL_rmse:.3f}')
# ax.scatter(apar[idx_day], ql_fit, color='red', marker = '.', label=f'Fitted parameters: alpha={alpha_ql:.3f}, beta={beta_ql:.5f}, hf_par={hf_par_fit:.1f}, R²={r_squared_qL:.3f}')
# ax.plot(par_range, ql_dir, color='blue', linestyle='--', label='Typical qL-PAR curve (dir)')
# ax.plot(par_range, ql_dif, color='green', linestyle='--', label='Typical qL-PAR curve (dif)')
cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
cbar.set_label('Air Temperature (°C)')
ax[0].set_xlabel('PAR (µmol/m2/s)')
ax[0].set_xlim([0, 2100])
ax[0].set_ylim([0, 2])
ax[0].set_ylabel('qL (unitless)')
# ax[0].legend()
# ax[0].set_title(f'qL vs PAR for DOY 150-250 in {yearstr}')

ax[1].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
           c = df_matched.loc[idx_day,'PAR_diffuse_fraction'], cmap='viridis', s=20, edgecolor='none')
# ax[1].scatter(par_demo, qL_fitted, s=20, edgecolor='k', facecolor='none', label = f'Fitted parameters: a={popt[0]:.3f}, b={popt[1]:.5f}')
cbar = plt.colorbar(ax[1].collections[0], ax=ax[1])
cbar.set_label('Diffuse Fraction')
ax[1].set_xlabel('PAR (µmol/m2/s)')
ax[1].set_xlim([0, 2100])
ax[1].set_ylim([0, 2])
ax[1].set_ylabel('qL (unitless)')
# ax[1].legend()
# ax[1].set_title(f'qL vs PAR for DOY 150-250 in {yearstr}')

ax[2].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
           c = df_matched.loc[idx_day,'DOY_UTC'], cmap='viridis', s=20, edgecolor='none')
# ax[1].scatter(par_demo, qL_fitted, s=20, edgecolor='k', facecolor='none', label = f'Fitted parameters: a={popt[0]:.3f}, b={popt[1]:.5f}')
cbar = plt.colorbar(ax[2].collections[0], ax=ax[2])
cbar.set_label('Day of Year (DOY)')
ax[2].set_xlabel('PAR (µmol/m2/s)')
ax[2].set_xlim([0, 2100])
ax[2].set_ylim([0, 2])
ax[2].set_ylabel('qL (unitless)')
# ax[2].legend()
# ax[2].set_title(f'qL vs PAR for DOY 150-250 in {yearstr}')

# ax[3].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
#            c = df_matched.loc[idx_day,'Ta (°C)'], cmap='viridis', s=20, edgecolor='none')
# ax[0].scatter(par_demo, qL_fitted, s=20, edgecolor='k', facecolor='none', label = f'Fitted parameters: a={popt[0]:.3f}, b={popt[1]:.5f}')
ax[3].scatter(par_demo, qL_fitted1, s=20, edgecolor='none', c= 'r',label='fitted' + f' R²={r_squared:.3f}, RMSE={rmse:.3f}')
# ax.scatter(apar[idx_day], ql_fit, color='red', marker = '.', label=f'Fitted parameters: alpha={alpha_ql:.3f}, beta={beta_ql:.5f}, hf_par={hf_par_fit:.1f}, R²={r_squared_qL:.3f}')
# ax.plot(par_range, ql_dir, color='blue', linestyle='--', label='Typical qL-PAR curve (dir)')
# ax.plot(par_range, ql_dif, color='green', linestyle='--', label='Typical qL-PAR curve (dif)')
# cbar = plt.colorbar(ax[3].collections[0], ax=ax[3])
# cbar.set_label('Air Temperature (°C)')
ax[3].set_xlabel('PAR (µmol/m2/s)')
ax[3].set_xlim([0, 2100])
ax[3].set_ylim([0, 2])
ax[3].set_ylabel('qL (unitless)')
ax[3].legend()
# ax[3].set_title(f'qL vs PAR for DOY 150-250 in {yearstr}')
fig.tight_layout()
fig.savefig(os.path.join(savepath_figs, f'qL_vs_PAR_DOY_150_250_DOYcolor.jpg'), dpi=300)


plt.show()

# fig, ax = plt.subplots(5,1, figsize=(12, 10))
# ax[0].scatter(df_matched.loc[idx_day, 'DOY_UTC'], apar[idx_day], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
# ax[0].set_xlabel('Day of Year (UTC)')
# ax[0].set_ylabel('APAR (µmol/m2/s)')
# ax[1].scatter(df_matched.loc[idx_day, 'DOY_UTC'], sifyield[idx_day], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
# ax[1].set_xlabel('Day of Year (UTC)')
# ax[1].set_ylabel('SIFyield (unitless)')
# ax[2].scatter(df_matched.loc[idx_day, 'DOY_UTC'], sifpsii[idx_day], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
# ax[2].set_xlabel('Day of Year (UTC)')
# ax[2].set_ylabel('SIFpsiisim (FS)')
# ax[3].scatter(df_matched.loc[idx_day, 'DOY_UTC'], df_matched.loc[idx_day,'JaLAI'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
# ax[3].set_xlabel('Day of Year (UTC)')
# ax[3].set_ylabel('JaLAI')
# ax[4].scatter(df_matched.loc[idx_day, 'DOY_UTC'], ql[idx_day], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
# ax[4].set_xlabel('Day of Year (UTC)')
# ax[4].set_ylabel('qL (unitless)')
# fig.savefig(os.path.join(savepath_figs, f'0_qL_APAR_SIF_JaLAI_vs_DOY_{yearstr}.jpg'), dpi=300)

# fig, ax = plt.subplots(2,2,figsize=(8, 6))
# ax[0,0].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], sifpsii[idx_day], \
#            c = df_matched.loc[idx_day,'DOY_UTC'], 
#            cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0])
# ax[0,1].scatter(df_matched.loc[idx_day, 'SIF_3FLD'], sifpsii[idx_day], \
#            c = df_matched.loc[idx_day,'DOY_UTC'], 
#            cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1])
# ax[1,0].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], df_matched.loc[idx_day,'SIFPSIILAI'], \
#               c = df_matched.loc[idx_day,'DOY_UTC'], 
#               cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[1,0].collections[0], ax=ax[1,0])
# ax[1,1].scatter(df_matched.loc[idx_day, 'SIF_3FLD'], df_matched.loc[idx_day,'SIFPSIILAI'], \
#               c = df_matched.loc[idx_day,'DOY_UTC'], 
#               cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[1,1].collections[0], ax=ax[1,1])
# ax[0,0].set_xlabel('PAR (µmol/m2/s)')
# ax[0,0].set_ylabel('SIFsim (FS)', color='blue')
# ax[1,0].set_xlabel('PAR (µmol/m2/s)')
# ax[1,0].set_ylabel('SIFsim (Gu)', color='orange')
# ax[0,1].set_xlabel('SIF_3FLD')
# ax[0,1].set_ylabel('SIFsim (FS)', color='blue')
# ax[1,1].set_xlabel('SIF_3FLD')
# ax[1,1].set_ylabel('SIFsim (Gu)', color='orange')
# fig.savefig(os.path.join(savepath_figs, f'SIFsim_vs_PAR_DOY_150_250_DOYcolor.jpg'), dpi=300)

# for i in range(150, 251):
#     idx_day_i = idx & (df_matched['DOY_UTC'] >= i) & (df_matched['DOY_UTC'] < i+1)
#     if idx_day_i.sum() == 0:
#         continue
#     fig, ax = plt.subplots(2,1, figsize=(8, 6))
#     ax[0].scatter(df_matched.loc[idx_day_i, 'PAR (µmol/m2/s)'], ql[idx_day_i], \
#                c = df_matched.loc[idx_day_i,'hour_UTC']+df_matched.loc[idx_day_i,'minute_UTC']/30.0, 
#                cmap='viridis', s=20, edgecolor='none')
#     cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
#     cbar.set_label('Hour of Day (UTC)')
#     ax[0].set_xlabel('PAR (µmol/m2/s)')
#     ax[0].set_ylabel('qL (unitless)')
#     ax[0].set_ylim(0, 1)
#     ax[0].set_xlim(0, 2050)
#     ax[1].plot(df_matched.loc[idx_day_i,'DOY_UTC'], sifpsii[idx_day_i], marker = 'o', color = 'blue', label='SIFsim (FS)')
#     ax1 = ax[1].twinx()
#     ax1.plot(df_matched.loc[idx_day_i,'DOY_UTC'], df_matched.loc[idx_day_i,'PAR (µmol/m2/s)'], marker = 'o', color='orange', label='SIFsim (Gu)')
#     # # 再添加一个y轴来绘制SIFcanopy_760nm
#     # ax2 = ax[1].twinx()
#     # # 调整第三个y轴的位置，避免与第二个y轴重叠
#     # ax2.spines['right'].set_position(('outward', 60))  # 将第三个y轴向右移动60像素
#     # ax2.plot(df_matched.loc[idx_day_i,'DOY_UTC'], df_matched.loc[idx_day_i,'SIFcanopy_760nm'], marker = 'o', color='green', label='SIFcanopy_760nm (Gu)')

#     ax[1].set_xlabel('Hour of Day (UTC)')
#     ax[1].set_ylabel('SIFPSIILAI')
#     ax[1].legend(loc='upper left')
#     ax1.set_ylabel('SIFsim (FS)', color='blue')
#     ax1.legend(loc='upper right')
#     # ax2.set_ylabel('SIFcanopy_760nm (Gu)', color='green')
#     # ax2.legend(loc='center right')
    
#     fig.savefig(os.path.join(savepath_figs, f'qL_vs_PAR_DOY_{i}.jpg'), dpi=300)
#     plt.close(fig)
# plt.show()
