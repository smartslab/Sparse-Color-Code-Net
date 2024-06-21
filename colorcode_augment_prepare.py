import os
import glob
import math
import yaml
import random
import torch
import shutil
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from colorcode_util import base_transform, photometric_transform, geometric_transform

img_size = (480, 640)
crop_size = (int(img_size[0]*0.9),int(img_size[1]*0.9))

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: (img * 2.0) - 1.0)  # Embedded the scaling function
])

photometric_transfrom = transforms.Compose([
    transforms.RandomPosterize(bits=5,p=0.2),
    transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.02),
    transforms.RandomAdjustSharpness(sharpness_factor=0.7,p=0.2)
])

geometric_transform = transforms.Compose([
    transforms.RandomCrop(size=crop_size), 
    transforms.RandomPerspective(distortion_scale=0.3, p=1),
    transforms.RandomRotation(degrees=60),
    transforms.Resize(size=img_size)
])

geometric_transform_fillgray = transforms.Compose([
    transforms.RandomCrop(size=crop_size, pad_if_needed=True, padding_mode='constant', fill=(128,128,128)),
    transforms.RandomPerspective(distortion_scale=0.3, p=1, fill=(128,128,128)),
    transforms.RandomRotation(degrees=60, fill=(128,128,128)),
    transforms.Resize(size=img_size)
])

class DeterministicTransform:
    def __init__(self, transform):
        self.transform = transform
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
    
    def __call__(self, image):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        return self.transform(image)

def augment_images(list_imglist, num_multiply, list_target_dir):

    # if target_directory1 not exist create it
    for target_directory in list_target_dir:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    # Function to determine the starting index for naming new files
    def get_starting_index(target_directory):
        existing_files = glob.glob(os.path.join(target_directory, '*.jpg')) + glob.glob(os.path.join(target_directory, '*.png'))
        if not existing_files:
            return 0
        else:
            max_index = max(int(os.path.splitext(os.path.basename(f))[0]) for f in existing_files)
            return max_index + 1

    # Get the starting index for file naming
    list_fileidx = []
    for target_directory in list_target_dir:
        list_fileidx.append(get_starting_index(target_directory))
    for idx in range(len(list_fileidx)):
        if list_fileidx[idx] != list_fileidx[0]:
            print("Error: list_fileidx[idx] != list_fileidx[0]")
            exit()
    file_index = list_fileidx[0]

    # Function to save image
    def save_image(image, index, target_directory, ext):

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image)
            elif image.ndim == 3:
                image = Image.fromarray(image, 'RGB' if image.shape[2] == 3 else 'RGBA')
            else:
                raise ValueError("Unsupported array shape for image conversion")
        
        filename = os.path.join(target_directory, f"{index:06d}"+ext)

        if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
            image.save(filename, quality=100)
        else:
            image.save(filename)

    deterministic_geom_transform = DeterministicTransform(geometric_transform)
    deterministic_geom_transform_fillgray = DeterministicTransform(geometric_transform_fillgray)

    # Augmentation process
    for idx_multiply in range(num_multiply):
        for idx_img in range(len(list_imglist[0])):
        
            deterministic_geom_transform.set_seed((idx_img+1)*(idx_multiply+1))
            deterministic_geom_transform_fillgray.set_seed((idx_img+1)*(idx_multiply+1))

            if idx_multiply == 0:
                for idx_folder in range(len(list_folder)):
                    ext = ".jpg" if list_folder[idx_folder] == "rgb" else ".png"
                    image = Image.open(list_imglist[idx_folder][idx_img])
                    save_image(image, file_index, list_target_dir[idx_folder], ext)
                file_index += 1
            else:
                for idx_folder in range(len(list_folder)):
                    ext = ".jpg" if list_folder[idx_folder] == "rgb" else ".png"
                    image = Image.open(list_imglist[idx_folder][idx_img])
                    if list_folder[idx_folder] == "rgb":
                        transformed_image = deterministic_geom_transform(photometric_transfrom(image))
                    elif list_folder[idx_folder] == "symm_mask":
                        transformed_image = deterministic_geom_transform_fillgray(image)
                    else:
                        transformed_image = deterministic_geom_transform(image)
                    save_image(transformed_image, file_index, list_target_dir[idx_folder], ext)
                file_index += 1

def list_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']  # Add more extensions if needed
    image_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files


with open("config_obj.yaml", "r") as file:
    info_obj = yaml.safe_load(file)

list_folder = ["rgb","cc_origin","cc_aniso","cc_symm","cc_aniso_n_symm","symm_mask"]

for idx_obj_str in info_obj:
    dir_src1 = "/content/LINEMOD/base/"
    dir_src2 = "/content/LINEMOD_pvnet_render/base"
    dir_tgt = "/content/LINEMOD_full_train/base/"
    list_target_dir = []
    for folder in list_folder:
        list_target_dir.append(os.path.join(dir_tgt,idx_obj_str,folder))
    
    # 1 augment the original training data
    list_imglist = [[] for i in range(len(list_folder))]
    with open(os.path.join(dir_src1,idx_obj_str,"train.txt"), 'r') as file:
        for line in file:
            img_name = line.strip()
            for idx_folder in range(len(list_folder)):
                if list_folder[idx_folder] == "rgb":
                    list_imglist[idx_folder].append(os.path.join(dir_src1,idx_obj_str,list_folder[idx_folder], img_name+".jpg"))
                else:
                    list_imglist[idx_folder].append(os.path.join(dir_src1,idx_obj_str,list_folder[idx_folder], img_name+".png"))
    
    augment_images(list_imglist, 5, list_target_dir)

    # 2 augment the synthetic data
    list_imglist = [[] for i in range(len(list_folder))]
    for idx_folder in range(len(list_folder)):
        list_imglist[idx_folder] = list_image_files(os.path.join(dir_src2,idx_obj_str,list_folder[idx_folder]))
        list_imglist[idx_folder].sort()

    for idx_folder in range(1, len(list_folder)):
        if len(list_imglist[idx_folder]) != len(list_imglist[0]):
            print("Error: Number of images in"+list_folder[idx_folder]+"and rgb are not equal")
            exit()

    augment_images(list_imglist, 1, list_target_dir)

    print("Done: "+idx_obj_str+"")
    _ = 0

