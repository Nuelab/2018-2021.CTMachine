# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:54:07 2020

@author: nuela
"""


import imageio
import numpy as np
from matplotlib import pyplot as plt
from math import *

import cv2 
import os 
import glob 
# from stl import mesh
from scipy.ndimage import gaussian_filter
# import png
import math
from numpy import save
from PIL import ImageEnhance
from PIL import Image, ImageDraw

def read_raw(raw_img_path):
	# Number of Rows
	ROWS = int(2944/2)
	# Number of Columns  
	COLS = int(2352/2)
	raw_img = open(raw_img_path)  
	# Loading the input image
# 	print("... Load input raw image")
	img = np.fromfile(raw_img, dtype = np.uint16, count = ROWS * COLS)
# 	print("Dimension of the old image array: ", img.ndim)
# 	print("Size of the old image array: ", img.size)
	# Conversion from 1D to 2D array
	img.shape = (img.size // COLS, COLS)
	img = np.rot90(img,1)
	return img



# def adjust_gamma(image, gamma=0.45):
# 	# Xây dựng bảng chuyển đổi liên kết giá trị mức xám [0-255] với giá trị gamma đã được điều chỉnh
#     # Giá trị gamma mặc định bằng 1.0-
# 	InvGamma = 1.0 / gamma
#     # Tạo Lookuptable
# 	lut = np.array([((i/255.0)**InvGamma)*255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	# Trả lại giá trị tương ứng cho ảnh bằng cách sử dụng hàm LUT
# 	return image

# def adjust_image_gamma(image, gamma = 1):
#   image = np.power(image, gamma)
#   max_val = np.max(image.ravel())
#   image = image/max_val * 65536
#   image = image.astype(np.uint16)
#   return image


img_dir1 = (r"E:\B.T.Hung\Samples\Sample3") # Enter Directory of all images  
data_path1 = os.path.join(img_dir1,'*.raw') 
files1 = glob.glob(data_path1)
# img1 = read_raw(r"1.30.2021/Data/0000.raw")
# plt.imshow(img1,cmap="gray")
# plt.show()

data = [] 
for f1 in files1: 
    img = read_raw(f1)
    img =  gaussian_filter(img, sigma=1)                       # Apply Gaussian Filter
    img = -np.log(img/np.max(img))*int(2**16)
    img.astype("uint16")
    cv2.normalize(img, img, 0, int(2**16), cv2.NORM_MINMAX)  # Cân bằng sáng
    img_crop = img[:,1000:1472]
    # im_width = img_crop.shape[1]
    # im_height = img_crop.shape[0]
    # gain = 1
    # img_crop = cv2.resize(img_crop,(int(im_width/gain),int(im_height/gain)))
    data.append(img_crop)
img3D = np.array(data)

# save('E:\B.T.Hung\Samples\Sample3\Sample3.npy', img3D)



# img1 = img3D[1,:,:]
# plt.imshow(img1,cmap="gray")
# plt.show()

# with open('E:\CBCT_KC05\Test_Recon_Image\pin\Preprocessing_Projs\PinProjs.npy', 'wb') as f:
#     np.save(f, img3D)


# output_img = r"E:\CBCT_KC05\Test_Recon_Image\Tru_nhom\DAQ\SaveImg\Img"

# # conv_img = -np.log(img3D[i,:,:]/np.max(img3D[i,:,:]))

# for i in range (0,360):
#     img1 = gaussian_filter(img3D[i,:,:], sigma=0.5)
#     conv_img = -np.log(img1/np.max(img1))
#     conv_img = cv2.
#     filename = str(i)+'.png'
#     cv2.imwrite(os.path.join(output_img,filename),conv_img)

# with open('E:\CBCT_KC05\Test_Recon_Image\pin\SaveImg\Pin.npy', 'wb') as f:
#     np.save(f, img3D)

# img1 = img3D[1,:,:]
# # # img1 = gaussian_filter(img1, sigma=0.5)

# # max_of_img = np.max(img1)
# # for i in range(0,676):
# #     for j in range(0,676):
# #         img1[i,j] = -1000*math.log(1.0*(img1[i,j])/max_of_img) 
# plt.imshow(conv_img,"gray")
# plt.show()
# # hist = cv2.calcHist([img1],[0],None,[256],[0,256])
# # plt.hist(img1.ravel(),256,[5,256]); plt.show()
# # plt.show()

# # with open('foo_gray2.png', 'wb') as f:
# #     writer = png.Writer(width=img1.shape[1], height=img1.shape[0], bitdepth=16, greyscale=True)
# #     zgray2list = img1.tolist()
# #     writer.write(f, zgray2list)