import os
import numpy as np
import cv2
from PIL import Image
import rasterio
from tqdm import tqdm

#source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/img_dir/test'
#destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/img_dir/test'
source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/img_dir/train'
destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/img_dir/train'
os.makedirs(destination_root,exist_ok=True)

filename_imgs = os.listdir(source_root)

for img_path in tqdm(filename_imgs):
    img = cv2.imread(os.path.join(source_root,img_path)) # 1000,1000,3
    
    img1 = img[:250,:250,:]
    img2 = img[:250,250:500,:]
    img3 = img[:250,500:750,:]
    img4 = img[:250,750:1000,:]
    
    img5 = img[250:500,:250,:]
    img6 = img[250:500,250:500,:]
    img7 = img[250:500,500:750,:]
    img8 = img[250:500,750:1000,:]   
    
    img9 = img[500:750,:250,:]
    img10 = img[500:750,250:500,:]
    img11 = img[500:750,500:750,:]
    img12 = img[500:750,750:1000,:]    
    
    img13 = img[750:1000,:250,:]
    img14 = img[750:1000,250:500,:]
    img15 = img[750:1000,500:750,:]
    img16 = img[750:1000,750:1000,:] 
    
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_1' + '_rgb.png'), img1)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_2' + '_rgb.png'), img2)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_3' + '_rgb.png'), img3)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_4' + '_rgb.png'), img4)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_5' + '_rgb.png'), img5)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_6' + '_rgb.png'), img6)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_7' + '_rgb.png'), img7)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_8' + '_rgb.png'), img8)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_9' + '_rgb.png'), img9)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_10' + '_rgb.png'), img10)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_11' + '_rgb.png'), img11)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_12' + '_rgb.png'), img12)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_13' + '_rgb.png'), img13)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_14' + '_rgb.png'), img14)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_15' + '_rgb.png'), img15)
    cv2.imwrite(os.path.join(destination_root, img_path.split('_rgb.jp2')[0] + '_16' + '_rgb.png'), img16)    
    

#source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/dem_dir/test'
#destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/dem_dir/test'
source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/dem_dir/train'
destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/dem_dir/train'
os.makedirs(destination_root,exist_ok=True)
prefix = '_dem.tif'

filename_imgs = os.listdir(source_root)

for img_path in tqdm(filename_imgs):
    img = np.array(Image.open(os.path.join(source_root,img_path))) # 1000,1000
    
    img1 = img[:250,:250]
    img2 = img[:250,250:500]
    img3 = img[:250,500:750]
    img4 = img[:250,750:1000]
    
    img5 = img[250:500,:250]
    img6 = img[250:500,250:500]
    img7 = img[250:500,500:750]
    img8 = img[250:500,750:1000]   
    
    img9 = img[500:750,:250]
    img10 = img[500:750,250:500]
    img11 = img[500:750,500:750]
    img12 = img[500:750,750:1000]    
    
    img13 = img[750:1000,:250]
    img14 = img[750:1000,250:500]
    img15 = img[750:1000,500:750]
    img16 = img[750:1000,750:1000] 
    
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_1' + prefix), img1)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_2' + prefix), img2)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_3' + prefix), img3)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_4' + prefix), img4)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_5' + prefix), img5)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_6' + prefix), img6)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_7' + prefix), img7)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_8' + prefix), img8)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_9' + prefix), img9)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_10' + prefix), img10)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_11' + prefix), img11)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_12' + prefix), img12)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_13' + prefix), img13)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_14' + prefix), img14)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_15' + prefix), img15)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_16' + prefix), img16)
    
#source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/ann_dir/test'
#destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/ann_dir/test'
source_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_mmseg/ann_dir/train'
destination_root = '/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/ann_dir/train'
os.makedirs(destination_root,exist_ok=True)
prefix = '_seg.tif'

filename_imgs = os.listdir(source_root)

for img_path in tqdm(filename_imgs):
    img = np.array(Image.open(os.path.join(source_root,img_path))) # 1000,1000
    
    img1 = img[:250,:250]
    img2 = img[:250,250:500]
    img3 = img[:250,500:750]
    img4 = img[:250,750:1000]
    
    img5 = img[250:500,:250]
    img6 = img[250:500,250:500]
    img7 = img[250:500,500:750]
    img8 = img[250:500,750:1000]   
    
    img9 = img[500:750,:250]
    img10 = img[500:750,250:500]
    img11 = img[500:750,500:750]
    img12 = img[500:750,750:1000]    
    
    img13 = img[750:1000,:250]
    img14 = img[750:1000,250:500]
    img15 = img[750:1000,500:750]
    img16 = img[750:1000,750:1000] 
    
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_1' + prefix), img1)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_2' + prefix), img2)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_3' + prefix), img3)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_4' + prefix), img4)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_5' + prefix), img5)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_6' + prefix), img6)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_7' + prefix), img7)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_8' + prefix), img8)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_9' + prefix), img9)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_10' + prefix), img10)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_11' + prefix), img11)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_12' + prefix), img12)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_13' + prefix), img13)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_14' + prefix), img14)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_15' + prefix), img15)
    cv2.imwrite(os.path.join(destination_root, img_path.split(prefix)[0] + '_16' + prefix), img16)
    

#import matplotlib.pyplot as plt
#rgb = np.array(Image.open('/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/img_dir/train/371_5712_10_rgb.png')) # 1000,1000,3
#dem = np.array(Image.open('/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/dem_dir/train/371_5712_10_dem.tif')) # 1000,1000
#mask = np.array(Image.open('/p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/ann_dir/train/371_5712_10_seg.tif')) # 1000,1000
#plt.subplot(1,3,1)
#plt.imshow(rgb)
#plt.subplot(1,3,2)
#plt.imshow(dem)
#plt.subplot(1,3,3)
#plt.imshow(mask)