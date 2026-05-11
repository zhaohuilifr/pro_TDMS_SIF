import os
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import savemat
from datetime import datetime

# 时间转换
def parse_time_to_matlab_datenum(t_str):
    # 匹配 '2024-07-19T05:12:01.289000'
    # %Y: 四位年份, %m: 月份, %d: 日期, %H: 小时, %M: 分钟, %S: 秒, %f: 微秒
    dt = datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%S.%f')
    # 转换为 MATLAB 兼容的 datenum 格式（可选）
    return dt.timestamp() / 86400.0 + 719529.0

def read_sif_file(path, fnname, metaI, ccalib):
    # 1. 加载 TDMS 文件
    tdms_path = os.path.join(path, "RAW", f"{fnname}.tdms")
    # tdms_file = TdmsFile.read(tdms_path)

    try:
        # 使用 read 模式打开，如果文件头损坏会抛出 ValueError
        tdms_file = TdmsFile.read(tdms_path)
    except ValueError as e:
        print(f"跳过损坏的文件 {fnname}.tdms: {e}")
        return # 提前结束，不生成损坏的 .mat 文件
    except Exception as e:
        print(f"读取 {fnname}.tdms 时发生未知错误: {e}")
        return
    
    # 获取所有组名（排除首尾非数据组，模拟 MATLAB 的 lacq(1)=[] 和 lacq(end)=[])
    all_groups = tdms_file.groups()
    lacq = [group.name for group in all_groups[1:-1]]
    
    # 2. 分类数据 (HR/LR 和 S/R)
    lLR_S = [n for n in lacq if not n.startswith('HR') and not n.endswith('REF')]
    lLR_R = [n for n in lacq if not n.startswith('HR') and n.endswith('REF')]
    lHR_S = [n for n in lacq if n.startswith('HR') and not n.endswith('REF')]
    lHR_R = [n for n in lacq if n.startswith('HR') and n.endswith('REF')]
    
    if len(lLR_S) == 0 or len(lLR_R)==0:
        print('no LR data found in file, skipping...')
        return
    
    mat_mapping = {
        'LR_S': lLR_S, 'LR_R': lLR_R, 
        'HR_S': lHR_S, 'HR_R': lHR_R
    }
    
    results = {}
    get_v = lambda k: metaI[k].values[0] if isinstance(metaI[k], (pd.Series, pd.DataFrame)) else metaI[k]

    # 3. 核心循环处理
    for nmat, list_groups in mat_mapping.items():
        wl = ccalib[f"{nmat[:2]}_WL"]
        wl_mask = (wl >= 400) & (wl <= 850)

        num_pix = get_v('sp_end') - get_v('sp_start') + 1
        tdata = np.full((num_pix, len(list_groups), 2), np.nan) # (Pixels, Groups, [White, Dark])
        all_meta = []
        
        for j, group_name in enumerate(list_groups):
            group = tdms_file[group_name]
            # 获取组属性 (Metadata)
            props = {k: v for k, v in group.properties.items()}
            # 时间解析
            props['Time_start'] = parse_time_to_matlab_datenum(str(props['Time_start']))
            props['Time_end'] = parse_time_to_matlab_datenum(str(props['Time_end']))
            # time_start = parse_time_to_matlab_datenum(str(props['Time_start']))
            # time_end = parse_time_to_matlab_datenum(str(props['Time_end']))
            
            # 区分暗电流帧(D)和信号帧(R)
            channels = group.channels()
            ldark = [ch for ch in channels if ch.name.startswith('D')]
            lwhite = [ch for ch in channels if ch.name.startswith('R')]
            
            # 数据读取辅助函数
            def get_data(ch_list):
                raw = np.array([ch.data.astype(float) for ch in ch_list]).T # (TotalPixels, Frames)
                sp = raw[get_v('sp_start')-1 : get_v('sp_end'), :] # MATLAB 1-based to 0-based
                dp = raw[get_v('dp_start')-1 : get_v('dp_end'), :]
                return sp, dp
            # 读取暗电流数据
            dark_sp, dark_dp = get_data(ldark)
            # 读取信号数据
            white_sp, white_dp = get_data(lwhite)

            # 计算屏蔽像素统计
            # 使用ddof=1计算样本标准差，保持与MATLAB std(x, 0)一致
            dp_dark_stats = np.array([np.nanmean(dark_dp, axis=0), np.nanstd(dark_dp, axis=0, ddof=1)]) # (2, nFrames)
            dp_white_stats = np.array([np.nanmean(white_dp, axis=0), np.nanstd(white_dp, axis=0, ddof=1)])

            # 4. 质量控制 (QC)
            # sun_var
            avg_dark_vec = np.nanmean(dark_sp, axis=1, keepdims=True)
            ssp = white_sp - avg_dark_vec
            ssp_f = ssp[wl_mask, :]
            sun_var = np.nanstd(np.nansum(ssp_f, axis=0) / np.nansum(np.nanmean(ssp_f, axis=1)), ddof=1)

            # 坏像素筛选
            ss_exp_dark = np.sum(dark_sp <= (dp_dark_stats[0] + 5*dp_dark_stats[1]), axis=0) / num_pix
            white_f = white_sp[wl_mask, :]
            ss_exp_white = np.sum(white_f <= (dp_white_stats[0] + 5*dp_white_stats[1]), axis=0) / white_f.shape[0]
            ss_sat_white = np.sum(white_sp > get_v('dyn_max'), axis=0) / num_pix

            # 过滤逻辑 (按 MATLAB if 逻辑)
            is_bad_white = (ss_exp_white > 0.01) | (ss_sat_white > 0.0015)
            is_bad_dark = ss_exp_dark < 0.99
            
            # 这里的逻辑必须和 MATLAB 保持一致：如果有好帧，则剔除坏帧；如果全坏，则保留。
            clean_white = white_sp[:, ~is_bad_white] if np.sum(~is_bad_white) > 0 else white_sp
            clean_dark = dark_sp[:, ~is_bad_dark] if np.sum(~is_bad_dark) > 0 else dark_sp
            
            # 更新元数据
            props.update({
                'sun_var': sun_var,
                'nbdark_sig': np.sum(is_bad_dark),
                'nbsp_nosig': np.sum(ss_exp_white > 0.01),
                'nbsp_bad': np.sum(is_bad_white),
                'prcdarkpix_sig': 1 - np.nanmean(ss_exp_dark),
                'prcsppix_nosig': np.nanmean(ss_exp_white),
                'prcsppix_sat': np.nanmean(ss_sat_white),
                'dark_dpix_mean': np.nanmean(dp_dark_stats[0, ~is_bad_dark]) if np.any(~is_bad_dark) else np.nanmean(dp_dark_stats[0]),
                'dark_dpix_std': np.nanmean(dp_dark_stats[1, ~is_bad_dark]) if np.any(~is_bad_dark) else np.nanmean(dp_dark_stats[1]),
                'sp_dpix_mean': np.nanmean(dp_white_stats[0, ~is_bad_white]) if np.any(~is_bad_white) else np.nanmean(dp_white_stats[0]),
                'sp_dpix_std': np.nanmean(dp_white_stats[1, ~is_bad_white]) if np.any(~is_bad_white) else np.nanmean(dp_white_stats[1]),
                'idx': float(j + 1)
            })

            all_meta.append(props)
            tdata[:, j, 0] = np.nanmean(clean_white, axis=1)
            tdata[:, j, 1] = np.nanmean(clean_dark, axis=1)

        results[nmat] = tdata
        # 模拟 MATLAB table: 将 dict 列表转为 record 数组
        results[f"meta_{nmat}"] = pd.DataFrame(all_meta).to_records(index=False)

    # 6. 保存为 .mat 文件
    save_path = os.path.join(path, 'PROCESSED', 'L0', f"{fnname}.mat")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为 scipy.io 可保存的字典
    savemat(save_path, results, format='5', do_compression=True) # MATLAB 兼容的 .mat 文件，需要使用decompression=False以避免兼容性问题
    
    print(f"Saved processed data to {save_path}")