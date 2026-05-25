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
        
        # # 6. 计算该轮廓的最大外接矩形坐标
        # # x, y 是矩形左上角坐标；w, h 是矩形的宽和高
        # x, y, w, h = cv2.boundingRect(max_contour)
        
        # # 适当向外扩大一点边缘（比如上下左右扩 10 个像素），防止裁剪得太死
        # padding = 10
        # img_h, img_w, _ = image.shape
        # x_new = max(0, x - padding)
        # y_new = max(0, y - padding)
        # w_new = min(img_w - x_new, w + 2 * padding)
        # h_new = min(img_h - y_new, h + 2 * padding)
        
        # # 7. 提取 ROI（利用 NumPy 切片）
        # roi = image[y_new:y_new+h_new, x_new:x_new+w_new]
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

def read_roi_tif(image_path):
    with open(image_path, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    
    # if image is None:
    #     print("Error: Could not load image.")
    # else:
    #     print("Image loaded successfully.")
    
    return image

def calculate_gcc(image_path, image_roi):
    image_jpg = read_jpg(image_path)
    if image_jpg is None:
        print("Error: Could not load image.")
        return np.nan, np.nan
    # r, g, b = cv2.split(image_jpg)  # 分离RGB通道
    ## ！！！如何用默认的int8，sum(r,g,b)会溢出，导致计算gcc错误 ！！！！

    r = image_jpg[image_roi == 255, 2].astype(np.int32)  # 红色通道  
    g = image_jpg[image_roi == 255, 1].astype(np.int32)  # 绿色通道
    b = image_jpg[image_roi == 255, 0].astype(np.int32)  # 蓝色通道

    gcc = g / (r + g + b)  # 计算GCC

    gcc_mean = np.nanmean(gcc)  # 计算ROI区域的GCC平均值
    gcc_std = np.nanstd(gcc)  # 计算ROI区域的GCC标准差

    # r_roi = np.nanmean(np.nanmean(image_jpg[image_roi == 255, 2]))  # 红色通道
    # g_roi = np.nanmean(np.nanmean(image_jpg[image_roi == 255, 1]))  # 绿色通道
    # b_roi = np.nanmean(np.nanmean(image_jpg[image_roi == 255, 0]))  # 蓝色通道
    # # if r_roi + g_roi + b_roi == 0:
    # #     return np.nan  # 避免除以零的情况
    # # return g_roi / (r_roi + g_roi + b_roi)
    # gcc_mean = g_roi / (r_roi + g_roi + b_roi)  # 计算ROI区域的GCC平均值

    return gcc_mean, gcc_std

# 提取时间戳
def extract_datetime_from_filename(filename):
    time_str = filename.split('_')[2].replace('.jpg', '') 
    # 格式化为 datetime 对象
    time_format = "%Y%m%d%H%M%S"  # 时间格式：YYYYMMDDHHMMSS
    timestamp = datetime.strptime(time_str, time_format)
    return timestamp

if __name__ == '__main__':
    # %% extract footprint
    # path_footprint = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2025\2.jpg'
    # get_footprint(path_footprint, save_path=path_footprint.replace('2.jpg', 'microlidar_roi_mask.tif'))

    # %% calculate gcc
    # datapath = r'E:\Datahub\Barbeau\Data_Camera\CameraMicroLidar\2025'
    # savepath = r'E:\Datahub\Barbeau\Data_Camera\GCC'
    # if not os.path.exists(os.path.join(savepath, '2025')):
    #     os.makedirs(os.path.join(savepath, '2025'))
    # roi_tif_path = r'E:\Datahub\Barbeau\Data_Camera\Footprint\2025\microlidar_roi_mask.tif'
    # roi_image = read_roi_tif(roi_tif_path)

    # df_all = []
    # for i_mom, folder_mon in enumerate(os.listdir(datapath)):
    #     path_mon = os.path.join(datapath, folder_mon)
    #     df_mon = []
    #     for i_day, folder_day in enumerate(os.listdir(path_mon)):
    #         path_day = os.path.join(path_mon, folder_day)
    #         # print(f"Image Path: {image}")
    #         for i_img, i_image in enumerate(os.listdir(path_day)):
    #             path_img = os.path.join(path_day, i_image)
    #             timestamp = extract_datetime_from_filename(i_image)
    #             gcc_value, gcc_std = calculate_gcc(path_img, roi_image)
    #             print(f"Timestamp: {timestamp}, GCC Mean: {gcc_value}, GCC Std: {gcc_std}")

    #             df_mon.append({
    #                 'datetime': timestamp,
    #                 'gcc_roi_mean': gcc_value,
    #                 'gcc_roi_std': gcc_std
    #             })
    #     df_all.extend(df_mon)
    #     # save df_mon
    #     df_mon = pd.DataFrame(df_mon)
    #     df_mon['datetime'] = df_mon['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    #     save_csv_path = os.path.join(savepath, '2025', f"{folder_mon}_gcc.csv")
    #     df_mon.to_csv(save_csv_path, index=False)
    #     print(f"GCC data for {folder_mon} saved to {save_csv_path}.")
    # # save df_all
    # df_all = pd.DataFrame(df_all)
    # df_all['datetime'] = df_all['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # save_csv_path_all = os.path.join(savepath, '2025', f"all_gcc_2025.csv")
    # df_all.to_csv(save_csv_path_all, index=False)

    # %% all_gcc_2025 cleaning
    df_all = pd.read_csv(r'E:\Datahub\Barbeau\Data_Camera\GCC\2025\all_gcc_2025.csv')
    idx_filter = (df_all['gcc_roi_std'] < 0.04) & (df_all['gcc_roi_mean']>0.25)
    # only daytime images
    df_all['hour'] = pd.to_datetime(df_all['datetime']).dt.hour
    idx_filter = idx_filter & (df_all['hour'] >= 8) & (df_all['hour'] <= 16)
    df_all_filtered = df_all[idx_filter]
    save_csv_path_all_filtered = r'E:\Datahub\Barbeau\Data_Camera\GCC\2025\all_gcc_2025_filtered.csv'
    df_all_filtered.to_csv(save_csv_path_all_filtered, index=False)