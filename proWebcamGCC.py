# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# 读取图片
def read_jpg(image_path):
    # 使用 cv2.imdecode 读取图片
    with open(image_path, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # if image is None:
    #     print("Error: Could not load image.")
    # else:
    #     print("Image loaded successfully.")
    return image

def save_roi_as_tif(image_path, roi_points, output_path="output.tif"):
    # 使用 cv2.imdecode 读取图片
    with open(image_path, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Could not load image.")
    else:
        print("Image loaded successfully.")

    # 创建与原始图像大小相同的二值掩码
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 将 ROI 区域填充为 1
    roi_points_np = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_points_np], 1)

    # 保存为 .tif 格式
    tif_image = Image.fromarray(mask * 255)  # 转换为 0 和 255 的二值图像
    tif_image.save(output_path)
    print(f"ROI mask saved as {output_path}")

def get_footprint(path_img, save_path="roi_mask.tif"):
    image = cv2.imread(path_img)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # HSV 的范围通常是：H [0, 180], S [0, 255], V [0, 255]
    # 针对图中这种较暗但纯正的蓝色激光/荧光，我们设定一个较宽但精准的蓝色区间：
    lower_blue = np.array([100,  50,  50])   # 蓝色的下限
    upper_blue = np.array([140, 255, 255])  # 蓝色的上限

    # 4. 创建掩膜（Mask）：符合蓝色范围的像素变白(255)，其余变黑(0)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 可选步骤：进行形态学开运算去噪（消除背景中可能存在的微小蓝色噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. 寻找蓝色区域的轮廓（Contours）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 找到面积最大的轮廓（即中央那块主要蓝色区域）
        roi_points = max(contours, key=cv2.contourArea)
        
        # 6. 计算该轮廓的最大外接矩形坐标
        # x, y 是矩形左上角坐标；w, h 是矩形的宽和高
        x, y, w, h = cv2.boundingRect(roi_points)
        
        # 适当向外扩大一点边缘（比如上下左右扩 10 个像素），防止裁剪得太死
        padding = 5
        img_h, img_w, _ = image.shape
        x_new = max(0, x - padding)
        y_new = max(0, y - padding)
        w_new = min(img_w - x_new, w + 2 * padding)
        h_new = min(img_h - y_new, h + 2 * padding)
        
        # 7. 提取 矩形区域的坐标点（左上、右上、右下、左下）
        roi = np.array([
            [x_new, y_new],
            [x_new + w_new, y_new],
            [x_new + w_new, y_new + h_new],
            [x_new, y_new + h_new]
        ], dtype=np.int32)

        # roi = [y_new:y_new+h_new, x_new:x_new+w_new]
        # 创建与原始图像大小相同的二值掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # 将 ROI 区域填充为 1
        roi_points_np = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points_np], 1)
        # 保存为 .tif 格式
        tif_image = Image.fromarray(mask * 255)  # 转换为 0 和 255 的二值图像
        tif_image.save(save_path.replace('.tif','_square.tif'))

        # 创建与原始图像大小相同的二值掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # 将 ROI 区域填充为 1
        roi_points_np = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points_np], 1)
        # 保存为 .tif 格式
        tif_image = Image.fromarray(mask * 255)  # 转换为 0 和 255 的二值图像
        tif_image.save(save_path)
        print(f"ROI mask saved as {save_path}")
    else:
        print("未在图像中检测到符合阈值的蓝色区域，请微调 lower_blue 或 upper_blue。")

def get_footprint2022(path_img, save_path="roi_mask.tif"):
    # 打开图像，手动选择ROI区域
    # 鼠标左键拖出一个矩形框（不是逐点点 4 个顶点），按 Enter/Space 确认，按 c 取消选择
    image = cv2.imread(path_img)
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    # 创建与原始图像大小相同的二值掩码
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # 将 ROI 区域填充为 1
    roi_points = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [roi_points], 1)
    # 保存为 .tif 格式
    tif_image = Image.fromarray(mask * 255)  # 转换为 0 和 255 的二值图像
    tif_image.save(save_path)
    print(f"ROI mask saved as {save_path}")

def read_roi_tif(image_path):
    with open(image_path, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    
    # if image is None:
    #     print("Error: Could not load image.")
    # else:
    #     print("Image loaded successfully.")
    
    return image

def calculate_gcc(image_path, image_roi, image_roi_sq):
    image_jpg = read_jpg(image_path)
    if image_jpg is None:
        print("Error: Could not load image.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # r, g, b = cv2.split(image_jpg)  # 分离RGB通道
    ## ！！！如何用默认的int8，sum(r,g,b)会溢出，导致计算gcc错误 ！！！！
    # 只计算ROI区域的像素值（使用浮点以允许 NaN 标记）
    raw_r = image_jpg[image_roi == 255, 2]# 红色通道
    raw_g = image_jpg[image_roi == 255, 1] # 原始绿色通道（用于检测饱和）
    raw_b = image_jpg[image_roi == 255, 0]# 蓝色通道
    r = raw_r.astype(np.float32)
    g = raw_g.astype(np.float32)
    b = raw_b.astype(np.float32)
    # 将饱和像素标为 NaN，避免对 gcc 计算的影响
    r[raw_r == 255] = np.nan
    g[raw_g == 255] = np.nan
    b[raw_b == 255] = np.nan
    gcc = g / (r + g + b)  # 计算GCC
    gcc_mean = np.nanmean(gcc)  # 计算ROI区域的GCC平均值
    gcc_std = np.nanstd(gcc)  # 计算ROI区域的GCC标准差

    # ROI square: 保留原始值用于饱和判断，再转换为浮点进行计算
    raw_r_sq = image_jpg[image_roi_sq == 255, 2]
    raw_g_sq = image_jpg[image_roi_sq == 255, 1]
    raw_b_sq = image_jpg[image_roi_sq == 255, 0]
    # 计算饱和像素数量与比例（在转换为 NaN 之前）
    sat_mask_sq = (raw_r_sq == 255) | (raw_g_sq == 255) | (raw_b_sq == 255)
    sat_count = np.sum(sat_mask_sq)
    flag_saturation = 1 if sat_count > 0 else 0
    flag_saturation_ratio = sat_count / 47300.0 / 3  # 220 * 215 的像素总数/3个通道
    # 转换为浮点并将饱和像素设为 NaN
    r_sq = raw_r_sq.astype(np.float32)
    g_sq = raw_g_sq.astype(np.float32)
    b_sq = raw_b_sq.astype(np.float32)
    r_sq[raw_r_sq == 255] = np.nan
    g_sq[raw_g_sq == 255] = np.nan
    b_sq[raw_b_sq == 255] = np.nan
    # 检验是否还有饱和像素残留（理论上不应该有，因为已经标记为 NaN 了）
    if np.any(r_sq == 255) or np.any(g_sq == 255) or np.any(b_sq == 255):
        print("Warning: There are still saturated pixels in the square ROI after masking.")
    gcc_sq = g_sq / (r_sq + g_sq + b_sq)  # 计算GCC
    gcc_sq_mean = np.nanmean(gcc_sq)  # 计算ROI区域的GCC平均值
    gcc_sq_std = np.nanstd(gcc_sq)  # 计算ROI区域的GCC标准差

    # GCC of the entire image（使用浮点以避免整型溢出和允许 NaN）
    r_all = image_jpg[:, :, 2].astype(np.float32)  # 红色通道
    g_all = image_jpg[:, :, 1].astype(np.float32)  # 绿色通道
    b_all = image_jpg[:, :, 0].astype(np.float32)  # 蓝色通道
    # 将饱和像素标为 NaN，避免对 gcc 计算的影响
    r_all[image_jpg[:, :, 2] == 255] = np.nan
    g_all[image_jpg[:, :, 1] == 255] = np.nan
    b_all[image_jpg[:, :, 0] == 255] = np.nan
    gcc_all = g_all / (r_all + g_all + b_all)  # 计算GCC
    gcc_all_mean = np.nanmean(gcc_all)  # 计算整个图像的GCC平均值
    gcc_all_std = np.nanstd(gcc_all)  # 计算整个图像的GCC标准差

    return gcc_mean, gcc_std, gcc_sq_mean, gcc_sq_std, gcc_all_mean, gcc_all_std, flag_saturation, flag_saturation_ratio

# 提取时间戳
def extract_datetime_from_filename(filename):
    time_str = filename.split('_')[2].replace('.jpg', '') 
    # 格式化为 datetime 对象
    time_format = "%Y%m%d%H%M%S"  # 时间格式：YYYYMMDDHHMMSS
    timestamp = datetime.strptime(time_str, time_format)
    return timestamp

if __name__ == '__main__':
    # %% extract footprint
    # path_footprint = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2022\1.jpg'
    # get_footprint2022(path_footprint, save_path=path_footprint.replace('1.jpg', 'microlidar_roi_mask.tif'))

    # path_footprint = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2023\1.jpg'
    # get_footprint(path_footprint, save_path=path_footprint.replace('1.jpg', 'microlidar_roi_mask.tif'))

    # path_footprint = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2024\2.jpeg'
    # get_footprint(path_footprint, save_path=path_footprint.replace('2.jpeg', 'microlidar_roi_mask.tif'))

    # path_footprint = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2025\2.jpg'
    # get_footprint(path_footprint, save_path=path_footprint.replace('2.jpg', 'microlidar_roi_mask.tif'))

    # 2026 : use the same footprint as 2025 (temporarily)
    # %% calculate gcc
    # years = ['2026']# '2025'， '2023','2024',
    # for year in years:
    #     datapath = r'E:\Datahub\Barbeau\Data_Camera\CameraMicroLidar\{year}'.format(year=year)
    #     savepath = r'E:\Datahub\Barbeau\Data_Camera\GCC'
    #     if not os.path.exists(os.path.join(savepath, year)):
    #         os.makedirs(os.path.join(savepath, year))
    #     roi_tif_path = r'E:\Datahub\Barbeau\Data_Camera\Footprint\{year}\microlidar_roi_mask.tif'.format(year=year)
    #     roi_image = read_roi_tif(roi_tif_path)
    #     roi_tif_path_sq = r'E:\Datahub\Barbeau\Data_Camera\Footprint\{year}\microlidar_roi_mask_square.tif'.format(year=year)
    #     roi_image_sq = read_roi_tif(roi_tif_path_sq)

    #     df_all = []
    #     for i_mom, folder_mon in enumerate(os.listdir(datapath)):
    #         path_mon = os.path.join(datapath, folder_mon)
    #         df_mon = []
    #         for i_day, folder_day in enumerate(os.listdir(path_mon)):
    #             path_day = os.path.join(path_mon, folder_day)
    #             # print(f"Image Path: {image}")
    #             for i_img, i_image in enumerate(os.listdir(path_day)):
    #                 if not i_image.lower().endswith(('.jpg', '.jpeg', '.png')):
    #                     print(f"Skip non-image file: {i_image}")
    #                     continue
    #                 path_img = os.path.join(path_day, i_image)
    #                 if not os.path.isfile(path_img) or os.path.getsize(path_img) == 0:
    #                     print(f"Skip invalid file: {path_img}")
    #                     continue
    #                 timestamp = extract_datetime_from_filename(i_image)
    #                 gcc_value, gcc_std, gcc_sq_mean, gcc_sq_std, gcc_all_mean, \
    #                     gcc_all_std, flag_saturation, flag_saturation_ratio = calculate_gcc(path_img, roi_image, roi_image_sq)
    #                 print(f"Timestamp: {timestamp}, GCC Mean: {gcc_value}, GCC Std: {gcc_std}")

    #                 df_mon.append({
    #                     'datetime': timestamp,
    #                     'gcc_roi_mean': gcc_value,
    #                     'gcc_roi_std': gcc_std,
    #                     'gcc_roi_sq_mean': gcc_sq_mean,
    #                     'gcc_roi_sq_std': gcc_sq_std,
    #                     'gcc_all_mean': gcc_all_mean,
    #                     'gcc_all_std': gcc_all_std,
    #                     'flag_saturation': flag_saturation,
    #                     'flag_saturation_ratio': flag_saturation_ratio
    #                 })
    #         df_all.extend(df_mon)
    #         # save df_mon
    #         df_mon = pd.DataFrame(df_mon)
    #         df_mon['datetime'] = df_mon['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    #         save_csv_path = os.path.join(savepath, year, f"{folder_mon}_gcc.csv")
    #         df_mon.to_csv(save_csv_path, index=False)
    #         print(f"GCC data for {folder_mon} saved to {save_csv_path}.")
    #     # save df_all
    #     df_all = pd.DataFrame(df_all)
    #     df_all['datetime'] = df_all['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    #     save_csv_path_all = os.path.join(savepath, year, f"all_gcc_{year}.csv")
    #     df_all.to_csv(save_csv_path_all, index=False)
    # %% for 2022
    years = ['2022']# '2025'， '2023','2024',
    for year in years:
        datapath = r'E:\Datahub\Barbeau\Data_Camera\CameraMicroLidar\{year}'.format(year=year)
        savepath = r'E:\Datahub\Barbeau\Data_Camera\GCC'
        if not os.path.exists(os.path.join(savepath, year)):
            os.makedirs(os.path.join(savepath, year))
        roi_tif_path = r'E:\Datahub\Barbeau\Data_Camera\Footprint\{year}\microlidar_roi_mask.tif'.format(year=year)
        roi_image = read_roi_tif(roi_tif_path)
        roi_tif_path_sq = r'E:\Datahub\Barbeau\Data_Camera\Footprint\{year}\microlidar_roi_mask_square.tif'.format(year=year)
        roi_image_sq = read_roi_tif(roi_tif_path_sq)

        df_all = []

        for i_img, i_image in enumerate(os.listdir(datapath)):
            if not i_image.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skip non-image file: {i_image}")
                continue
            path_img = os.path.join(datapath, i_image)
            if not os.path.isfile(path_img) or os.path.getsize(path_img) == 0:
                print(f"Skip invalid file: {path_img}")
                continue
            if "MicoL" in i_image:
                time_str = i_image.split(' ')[3] + i_image.split(' ')[2] + i_image.split(' ')[1]+ \
                    i_image.split(' ')[-1].replace('.jpg', '').split('h')[0] + i_image.split(' ')[-1].replace('.jpg', '').split('h')[1]
                # 格式化为 datetime 对象
                time_format = "%Y%m%d%H%M"  # 时间格式：YYYYMMDDHHMMSS
                timestamp = datetime.strptime(time_str, time_format)
            else:
                time_str = i_image.split('_')[3] + i_image.split('_')[2]
                # 格式化为 datetime 对象
                time_format = "%Y%m%d%H%M%S"  # 时间格式：YYYYMMDDHHMMSS
                timestamp = datetime.strptime(time_str, time_format)
                image_jpg = read_jpg(path_img)
                # scale roi_image to the same size as image_jpg
                if image_jpg is None:
                    print("Error: Could not load image.")
                    continue
                roi_image = scale_image(roi_image, image_jpg.shape[1], image_jpg.shape[0])
                roi_image_sq = scale_image(roi_image_sq, image_jpg.shape[1], image_jpg.shape[0])

            gcc_value, gcc_std, gcc_sq_mean, gcc_sq_std, gcc_all_mean, \
                gcc_all_std, flag_saturation, flag_saturation_ratio = calculate_gcc(path_img, roi_image, roi_image_sq)
            print(f"Timestamp: {timestamp}, GCC Mean: {gcc_value}, GCC Std: {gcc_std}")

            df_all.append({
                'datetime': timestamp,
                'gcc_roi_mean': gcc_value,
                'gcc_roi_std': gcc_std,
                'gcc_roi_sq_mean': gcc_sq_mean,
                'gcc_roi_sq_std': gcc_sq_std,
                'gcc_all_mean': gcc_all_mean,
                'gcc_all_std': gcc_all_std,
                'flag_saturation': flag_saturation,
                'flag_saturation_ratio': flag_saturation_ratio
            })
        # save df_all
        df_all = pd.DataFrame(df_all)
        df_all['datetime'] = df_all['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        save_csv_path_all = os.path.join(savepath, year, f"all_gcc_{year}.csv")
        df_all.to_csv(save_csv_path_all, index=False)
    # %% all_gcc_2025 cleaning
    # years = ['2023','2024','2026']  # '2025'， '2023','2024',
    # for year in years:
    #     df_all = pd.read_csv(r'E:\Datahub\Barbeau\Data_Camera\GCC\{year}\all_gcc_{year}.csv'.format(year=year))
    #     idx_filter = (df_all['gcc_roi_sq_std'] < 0.045) & (df_all['gcc_roi_sq_mean']>0.25) & (df_all['gcc_roi_sq_mean']<0.5)
    #     # only daytime images
    #     df_all['hour'] = pd.to_datetime(df_all['datetime']).dt.hour
    #     idx_filter = idx_filter & (df_all['hour'] >= 8) & (df_all['hour'] <= 16)
    #     df_all_filtered = df_all[idx_filter]
    #     save_csv_path_all_filtered = r'E:\Datahub\Barbeau\Data_Camera\GCC\{year}\all_gcc_{year}_filtered.csv'.format(year=year)
    #     df_all_filtered.to_csv(save_csv_path_all_filtered, index=False)