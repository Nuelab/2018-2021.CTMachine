from __future__ import division

import numpy as np
from os.path import join
import astra
import cv2
from time import time
import os 
import glob 
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import itk

'''
        DEFINE USER FUNCTION TO READ THE RAW IMAGE
'''
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


for step in range(0,1,1):
    astra.set_gpu_index([0])                                                   # Specify CPU ID to reconstruc image
    # '''
    #     READ ALL THE RAW IMAGES CONTAINDE IN THE SOURCE FOLDER
    # '''
    # # Start time
    print("Initiate cycle",step+1)
    print("")
    print("=> Read images")
    start_time = time()
    img_dir1 = (r"E:\B.T.Hung\Samples\Standard_Mau_Tru_Bac_Nhom") # Enter Directory of all images  
    data_path1 = os.path.join(img_dir1,'*.raw') 
    files1 = glob.glob(data_path1)
    data = [] 
    for f1 in files1: 
        img = read_raw(f1)
           
        img =  gaussian_filter(img, sigma=1)                       # Apply Gaussian Filter
        # img = ndimage.median_filter(img, size=3)
        img = -np.log(img/np.max(img))*int(2**16)
        img.astype("uint16")
        cv2.normalize(img, img, 0, int(2**16), cv2.NORM_MINMAX)  # Cân bằng sáng
        # img = np.rot90(img,2) 
        height = img.shape[0]
        weight = img.shape[1]
        img_crop = img[step:height,0:weight]
        data.append(img_crop)
   
    img3D = np.array(data)
    # img3D = np.load(r'E:\B.T.Hung\Samples\Standard_Mau_Airfoild\Reconstructed_image\projections.npy')
    end_time = time()
    print("=> Read images done. Time=",end_time-start_time)
    print("")
    
    print("=> Start reconstruction")
    start_time = time()
    output_dir = r"E:\B.T.Hung\Samples\Standard_Mau_Tru_Bac_Nhom\Reconstructed_image"
    data_3D = img3D
    # Configuration.
    SDD = 993                                     # Source to Detector Distance [mm] 
    SOD = 841.1                                   # Source to Object Distance   [mm]
    ODD = SDD - SOD                               # Object to Detector Distance [mm] 
    detector_pixel_size = 0.099                   # Pixel size                  [mm] 
    detector_rows = data_3D.shape[1]              # Vertical size of detector   [pixels].
    detector_cols = data_3D.shape[2]              # Horizontal size of detector [pixels].
    #num_of_projections = data_3D.shape[0]
    num_of_projections = data_3D.shape[0]
    # Creating number of projections array
    angles = np.linspace(0, 2*np.pi, num=num_of_projections, endpoint=False)
    
    # Load projections.
    print ("=> Loading Projection ...")
    projections = np.zeros((detector_rows, num_of_projections, detector_cols))
    for i in range(num_of_projections):
      img = data_3D[i,:,:]
    #  img = apply_filter(img, filter_name='ram_lak')
      
      projections[:, i, :] = img                       # Lưu hình chiếu vào không gian hình chiếu
    
    # Khởi tạo hình học hình chiếu, dữ liểu trả về dạng dict bao gồm các thông số: 
    #    - Kích thước đầu dò
    #    - Kích thước pixel
    #    - Mảng lưu giá trị các góc chiếu
    #    - Kiểu hình học chiếu (song song, rẻ quạt hoặc nón)
    print ("=> Create Project Geometry")
    proj_geom = astra.create_proj_geom('cone',  1, 1, 
                                        detector_rows, detector_cols, angles, 
                                        SOD/detector_pixel_size, ODD/detector_pixel_size)
    
    # Khởi tạo biến lưu trữ hình chiếu (ID - kiểu int), đây là một sinogram 3D với thông số
    # hình học chiếu lấy từ biến proj_geom. Biến projections_id được dùng để định danh 
    # hình chiếu và được sử dụng để cấu hình thuật toán chiếu ngược sau này.
    projections_id = astra.data3d.create('-sino', proj_geom, projections)
    
    # Khởi tạo không gian tái tạo vật thể
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    
    # Khởi tạo biến lưu trữ ID của không gian tái tạo 3D, với kích thước của không gian 
    # được lưu trong biến vol_geom, giá trị ban đầu của mỗi điểm anh trong không gian
    # được khai báo bằng 0
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    
    # Biến alg_cfg có kiểu dict, dùng để lưu trông số cấu hình cho thuật toán tái tạo
    alg_cfg = astra.astra_dict('FDK_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    #alg_cfg['GPUindex'] = 0
    
    # Khởi tạo thuật toán từ biến alg_cfg
    algorithm_id = astra.algorithm.create(alg_cfg)
    
    # Chạy thuật toán, hàm này có hai biến đầu vào:
    #    - algorithm_id: ID của thuật toán, kiểu int
    #    - iterations vòng lặp cho thuật toán đại số, mặc định bằng 1 với thuật toán chiếu ngược
    iter_val = 1
    print("=> Running Algorithm ...")
    astra.algorithm.run(algorithm_id,iter_val)
    
    # Thu dữ liệu tái tạo, dữ liệu trả về có dạng numpy.ndarray
    reconstruction = astra.data3d.get(reconstruction_id)
    
    # Loại bỏ giá trị âm của dữ liệu tái tạo gây ra do noise
    # Định dạng lại dữ liệu ra có kiểu 8 bit
    reconstruction[reconstruction < 0] = 0
    reconstruction /= np.max(reconstruction)
    reconstruction = np.round(reconstruction * 65535).astype(np.uint16)
    
    # reconstruction = np.round(reconstruction * 255).astype(np.uint8)
    # End time
    end_time = time()-start_time
    
    # Lưu dữ liệu ra file
    # print ("=> Saving reconstructed image... ") 
    # print ("=> Saving result ...")
    # for i in range(detector_cols):
    # im = reconstruction[:, 736, :]
    # im = np.flipud(im)
    # cv2.imwrite(join(output_dir, 'recoH%04d.png' % step), im)
    
    # im = reconstruction[588, :, :]
    # im = np.flipud(im)
    # cv2.imwrite(join(output_dir, 'recoV%04d.png' % step), im)  
    image = itk.GetImageFromArray(reconstruction)
    itk.imwrite(image, 'recondat.nrrd')
    # print ("=> Saving 3D array data... ")
    # np.save('E:\B.T.Hung\Samples\Standard_Mau_Den_Pin\Reconstructed_image\data.npy',reconstruction)
    
    # Cleanup.
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)
    
    print ("=> FINISH <=")
    print("Running time "+str(iter_val)+"  = ", end_time)
