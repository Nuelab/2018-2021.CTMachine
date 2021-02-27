# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:22:35 2020

@author: buiha
"""

import multiprocessing 
from multiprocessing import Pool
import numpy as np
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from scipy import signal
from functools import partial
import os
import astra
from os.path import join
from time import time
import os 
import glob
import itk 


def filter_generator(size, filter_name="ram_lak",t=0.1):
    """

    Parameters
    ----------
    size : (int)
        size of filter.
    filter_name : (str), 
        The name of filter. The default is "ram_lak".
    t : (float), 
        Sampling interver (mm). The default is 0.049.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    W = 1/(2*t)
    size = int(size*2)
    ram_filter = np.zeros((size))
    for i in range (size):
        ram_filter[i] = abs(float(i-size/2))/(size/2)*W
    if filter_name == "ram_lak":
        fourier_filter = ram_filter.max() - ram_filter
    # Shepp Logan window
    elif filter_name == "shepp_logan":
        # Start from first element to avoid divide by zero
        fourier_filter = ram_filter.max() - ram_filter
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    # Cosine window
    elif filter_name == "sine":
        omeg = np.linspace(0,np.pi,size)
        sine = np.sin(omeg)
        fourier_filter = ram_filter*sine
    # Sine window
    elif filter_name == "cosine":
        omeg = np.linspace(0,np.pi,size)
        cosine = np.cos(omeg-np.pi/2)
        fourier_filter = ram_filter*cosine
    # Hamming window
    elif filter_name == "hamming":
        hamming = np.hamming(size)
        fourier_filter = ram_filter*hamming
    # Hanning window
    elif filter_name == "hann":
        hanning = np.hanning(size)
        fourier_filter = ram_filter*hanning
    # Parzen window
    elif filter_name == "parzen":
        pazen = signal.parzen(size)
        fourier_filter = ram_filter*pazen
    # Flattop window
    elif filter_name == "flattop":
        a0, a1, a2, a3, a4 = 0.21, 0.41, 0.28, 0.08, 0.007
        omeg = np.linspace(0,np.pi,size)
        flattop = a0 - a1*np.cos(2*omeg) + a2*np.cos(4*omeg) - a3*np.cos(6*omeg) + a4*np.cos(8*omeg)
        # flattop = signal.flattop(size)
        fourier_filter = ram_filter*flattop
    # Gaussian window
    elif filter_name == "gauss":
        gauss = signal.gaussian(size,200)
        fourier_filter = ram_filter*gauss
    # Butterworth window
    elif filter_name == "butter":
        freq = np.linspace(-5,5,size)
        butter1 = 1/np.sqrt((1+(freq/0.4)**(12)))
        fourier_filter = ram_filter*butter1
    elif filter_name == "None":
        fourier_filter = np.zeros((size))
        fourier_filter[:] = 1
    filter_types = ('ram_lak', 'shepp_logan', 'cosine', 'sine', 'hamming', 'hann',
                    'parzen', 'flattop', 'gauss', 'butter','None')
    if filter_name not in filter_types:
        raise ValueError("Hàm lọc không xác định: %s" % filter_name)

    return fourier_filter#[np.newaxis,:]

def apply_filter(img, filter_name='ram_lak'):
    """ Apply filter for 2D projection
    """
    height = img.shape[0]
    weight = img.shape[1]
    img_shape = img.shape[1]
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    projection_size_padded = int(2 * img_shape)
    pad_width = ((0, 0), ((projection_size_padded - img_shape)//2, 
                          (projection_size_padded - img_shape)//2))
    # print(pad_width)
    img = np.pad(img, pad_width, mode='constant', constant_values=0)
    # Apply filter in Fourier domain
    fourier_filter = filter_generator(img_shape,filter_name)
    fourier_filter = np.pad(fourier_filter,(projection_size_padded - int(2*img_shape))//2,
                            mode='constant', constant_values=0)
    fourier_filter = fourier_filter[np.newaxis,:]
    projection = fft(img, axis=1) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=1)[:img.shape[0], :])
    # Crop to Ogirinal Image
    nweight = img.shape[1]
    img_crop = radon_filtered[0:height,(nweight-weight)//2:(nweight+weight)//2]  
    img_crop[:,0:5] = 0
    img_crop[:,img_crop.shape[1]-5:img_crop.shape[1]-0] = 0
    return img_crop

def img_preprocessing(img):
    img =  gaussian_filter(img, sigma=1)
    # img = median_filter(img,11)
    img = -np.log(img/img.max())
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img

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

def read_allImages(path,option='*.raw'):
    data_path = os.path.join(path,option) 
    files = glob.glob(data_path)
    data = [] 
    for f in files: 
        img = read_raw(f)
        data.append(img)
    data = np.array(data)
    return data

""" ============================================================================================== """
""" ========================================= Main Program ======================================= """
""" ============================================================================================== """
if __name__ == '__main__':
    No_of_CPU = os.cpu_count()                                                 # Count number of CPU
    filter_names = ['ram_lak', 'shepp_logan', 'cosine', 'hamming', 'hann', 'butter','tukey',
                    'parzen', 'gauss', 'flattop', 'triangular', 'barlett-hann', 'blackman',
                    'nuttall', 'blackman_harris', 'blackman-nuttall', 'rsinogram',
                    'kaiser', 'lanczos', 'projection', 'sinogram', 'rprojection', 'None']
        
    print('Select filter in list: ', filter_names)
    print('filter_name = ')
    filter_name = input()                                                      # Select filter name
    
    if filter_name not in filter_names:
        raise ValueError("Wrong filter: %s" % filter_name)
    
    start = time()
    print("=> Loading projection...")
    # datapath = "E:\B.T.Hung\Samples\Standard_Mau_Tru_Nhom_Chop_2"
    output_dir = "E:\B.T.Hung\Samples\Standard_Mau_Dien_Thoai" 
    # data = read_allImages(r'E:\B.T.Hung\Samples\Danh_Gia_Do_On_Dinh_chu_trinh\CT1')  
    data = np.load(r'E:\B.T.Hung\Samples\Standard_Mau_Dien_Thoai\projections.npy')
    t_pre_start = time()
    print("=> Preprocessing Projections...")
    with Pool(No_of_CPU//4*3) as p:                                            # Using 3/4 number of cores for calculation
        result = np.asarray(p.map(img_preprocessing, data[0:data.shape[0],:,:]))
    t_pre_stop = time()
    
    del data   
    print ("=> Config Projection Geometry...")
    with Pool((No_of_CPU//4)*3) as p:                                            # Using 3/4 number of cores for calculation
        data_3D = np.asarray(p.map(partial(apply_filter,filter_name=filter_name),
                                    result[0:result.shape[1],:,:]))

    del result
    print ("=> Config Projection Geometry...")
    SDD = 993                                                                  # Source to Detector Distance [mm] 
    SOD = 841.1                                                                # Source to Object Distance   [mm]
    ODD = SDD - SOD                                                            # Object to Detector Distance [mm] 
    detector_pixel_size = 0.099                                                # Pixel size                  [mm] 
    detector_rows = data_3D.shape[1]                                           # Vertical size of detector   [pixels].
    detector_cols = data_3D.shape[2]                                           # Horizontal size of detector [pixels].
    num_of_projections = data_3D.shape[0]                                      # Lấy tự động số lượng hình chiếu
    angles = np.linspace(0, 2*np.pi, num=num_of_projections, endpoint=False)   # Tạo ma trận chiếu
    
    print ("=> Loading Projection ...")
    projections = np.zeros((detector_rows, num_of_projections, detector_cols),dtype=np.uint8)
    for i in range(num_of_projections):
        img = data_3D[i,:,:]
        projections[:, i, :] = img                                                              # Lấy thời gian ở lúc hiện tại
#    del img_data
    del data_3D

    print ("=> Create Project Geometry...")
    proj_geom = astra.create_proj_geom('cone',  1, 1, 
                                  detector_rows, detector_cols, angles, 
                                  SOD/detector_pixel_size, ODD/detector_pixel_size)
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                         detector_rows)
    print("=> Running Algorithm ...")
    bproj_id, reconstruction = astra.create_backprojection3d_gpu(projections, proj_geom, vol_geom)
    
    image = itk.GetImageFromArray(reconstruction)
    itk.imwrite(image,join(output_dir,'recondat.nrrd'))
    
    astra.data3d.delete(bproj_id)
    print("=> Reconstruction process finish in",(time()-start))