# -*- coding: utf-8 -*-

###
# remove the pulses at night from the data
###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

def validate_nopulse(df, df_cleaned, savepath):
    # print(f'Validating pulse removal, saving figure to {savepath}')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['DOY'], df['Fs'], color='r', label='Pulses', alpha=0.7)
    ax.plot(df_cleaned['DOY'], df_cleaned['Fs'], color='k', label='Cleaned Fs', alpha=1)
    doy_int = df['DOY'].astype(int)[100]
    ax.set_xlim(doy_int, doy_int + 1)  # Set x-axis limits to show one day
    ax.set_xlabel('DOY')
    ax.set_ylabel('Fs')
    ax.set_title('Validation of Pulse Removal')
    ax.legend()
    fig.savefig(savepath, dpi=300)
    plt.close()

def remove_pulses_night(df, t_pulse=[0, 0.041, 0.082, 0.123], duration=0.005):
    # remove the pulses at night from the data
    t = df['DOY'] - df['DOY'].astype(int)
    idx_pulse = ((t >= t_pulse[0]) & (t <= t_pulse[0] + duration)) | \
                ((t >= t_pulse[1]) & (t <= t_pulse[1] + duration)) | \
                ((t >= t_pulse[2]) & (t <= t_pulse[2] + duration)) | \
                ((t >= t_pulse[3]) & (t <= t_pulse[3] + duration))
    df_cleaned = df.copy()
    if len(idx_pulse) > 0:
        df_cleaned.loc[idx_pulse, 'Fs'] = np.nan
    return df_cleaned

t_pulse = [0, 0.041, 0.082, 0.123]
duration =  15/60.0/24 # remove 10 minutes from each pulse, in days


if __name__ == "__main__":
    # %% 设置路径
    root = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau'
    yearstr = '2025' # '2025' # '2022'
    lif_path = os.path.join(root, 'PROCESSED', yearstr, 'L2')
    lif_path_Daily = os.path.join(lif_path, 'Daily')
    lif_path_Yearly = os.path.join(lif_path, 'Yearly')
    save_path_daily = os.path.join(lif_path, 'Daily_nopulse')
    save_path_yearly = os.path.join(lif_path, 'Yearly_nopulse')
    if not os.path.exists(save_path_daily):
        os.makedirs(save_path_daily)
    if not os.path.exists(save_path_yearly):
        os.makedirs(save_path_yearly)
    validate_daily = os.path.join(lif_path, 'Daily_nopulse_validation')
    if not os.path.exists(validate_daily):
        os.makedirs(validate_daily)

    # loop all *.csv files in the lif_path_Daily and remove the pulses at night
    df_yearly = pd.DataFrame()
    for file in os.listdir(lif_path_Daily):
        if file.endswith('.csv'):
            # print(f'Processing file: {file}')
            df = pd.read_csv(os.path.join(lif_path_Daily, file))
            df_cleaned = remove_pulses_night(df, t_pulse=t_pulse, duration=duration)

            # validate the pulse removal
            # validate_nopulse(df, df_cleaned, os.path.join(validate_daily, file.replace('.csv', '_validation.jpg')))

            # save the cleaned dataframe to a new csv file
            df_yearly = pd.concat([df_yearly, df_cleaned], ignore_index=True)
            df_cleaned.to_csv(os.path.join(save_path_daily, file), index=False)
    df_yearly.to_csv(os.path.join(save_path_yearly, f'{yearstr}_LIF.csv'), index=False)