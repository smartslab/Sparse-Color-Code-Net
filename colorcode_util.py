import cv2
import sys
import os
import math
import random
import torch
import shutil
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

img_size = (480, 640)
crop_size = (int(img_size[0]*0.9),int(img_size[1]*0.9))

class ScaleTransform:
    def __call__(self, img):
        return (img * 2.0) - 1.0

base_transform = transforms.Compose([
    transforms.ToTensor(),
    ScaleTransform()  # Replacing the lambda function
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

def sobel_generator(num_channel=3):
    Gx3 = np.array([[0.0,0,0,0,0],[0,-1,-2,-1,0],[0,0,0,0,0],[0,1,2,1,0],[0,0,0,0,0]])*4
    Gx5 = np.array([[2.0,2,4,2,2],[1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2,-1,-1],[-2,-2,-4,-2,-2]])
    Gy3 = np.transpose(Gx3)
    Gy5 = np.transpose(Gx5)
    sobel = np.zeros((4,num_channel,5,5))
    for i in range(num_channel):
      sobel[0,i,:,:] = Gx3
      sobel[1,i,:,:] = Gy3
    #   sobel[2,i,:,:] = Gx5
    #   sobel[3,i,:,:] = Gy5
    sobel = torch.from_numpy(sobel).float()
    return sobel

def generate_mask_contour(sobel,img,thresh=7.5):
    # img_abs_diff1 = torch.abs(img[:,0,:,:]-img[:,1,:,:])*2
    # img_abs_diff2 = torch.abs(img[:,1,:,:]-img[:,2,:,:])*2
    # img_abs_diff3 = torch.abs(img[:,2,:,:]-img[:,0,:,:])*2

    # img = torch.cat((img, img_abs_diff1.unsqueeze(1), img_abs_diff2.unsqueeze(1), img_abs_diff3.unsqueeze(1)), dim=1)

    img_sobel = F.conv2d(img, sobel, padding="same")
    img_sobel_abssum = torch.abs(img_sobel).sum(dim=1, keepdim=True)
    
    # img_sobel_abssum_max, _ = torch.max(img_sobel_abssum.view(img.size(0), -1), dim=1, keepdim=True)
    # img_sobel_abssum_max = img_sobel_abssum_max.view(img.size(0), 1, 1, 1)
    # mask_contour = (img_sobel_abssum > (0.05 * img_sobel_abssum_max)).float()

    mask_contour = (img_sobel_abssum>thresh).float()

    return mask_contour

def random_affine_transform(mask, scale=(0,0), angle=(0,0), trans_r=(0,0)):

    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    batch_size, channel, h, w = mask.shape
    mask = mask.reshape(-1, h, w)

    scale = random.uniform(scale[0], scale[1])
    angle = random.uniform(angle[0], angle[1])
    shift_dx = random.choice([-1, 1])*random.uniform(trans_r[0], trans_r[1])*w
    shift_dy = random.choice([-1, 1])*random.uniform(trans_r[0], trans_r[1])*h

    mask = transforms.functional.affine(mask, angle=angle, translate=[shift_dx, shift_dy], scale=scale, shear=[0, 0],
                    interpolation=InterpolationMode.NEAREST)

    mask = mask.reshape(batch_size, channel, h, w)

    return mask

def save_image_from_tensor(img_tensor,img_path):
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
    img_tensor = (img_tensor+1)/2*255
    img = img_tensor.cpu().numpy()
    img = img.transpose((1,2,0))
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if os.path.exists(img_path):
        os.remove(img_path)
    cv2.imwrite(img_path, img)

def retain_valid_mask(masks_prob, morph_conv):

    k1, k2, k3 = 16, 8, 8

    # get the max pool with different threshold at different scale
    masks_pred_h = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k1, stride=k1)>0.9).float()
    masks_pred_m = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k2, stride=k2)>0.7).float()
    masks_pred_l = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k3, stride=k3)>0.5).float()
    
    # get the high thresh mask a dilate
    masks_pred_h = (morph_conv(masks_pred_h)>0.5).float()
    masks_pred_h = torch.nn.functional.interpolate(masks_pred_h, scale_factor=int(k1/k2))

    # find overlap of dilated high thresh mask and medium thresh mask, then dilate the overlapping region
    masks_pred_m = masks_pred_m*masks_pred_h
    masks_pred_m = (morph_conv(masks_pred_m)>0.5).float()
    masks_pred_m = torch.nn.functional.interpolate(masks_pred_m, scale_factor=int(k2/k3))

    # find overlap of dilated mid thresh mask and low thresh mask
    masks_pred_l = masks_pred_l*masks_pred_m

    masks_retain = masks_pred_l
    scale = k3

    return masks_retain, scale

def get_bbox(tensor, scale=1):

    if tensor.sum() == 0:
        bbox = None
    else:
        rows = torch.where(tensor.any(dim=1))[0]
        cols = torch.where(tensor.any(dim=0))[0]

        bbox = [int(cols.min()*scale), int(rows.min()*scale), int(cols.max()*scale), int(rows.max()*scale)]

    return bbox

def smooth_within_mask(img_tensor, kernel_size=7, sigma=1):

    # take all values from range (-1,1) to range(0,1)
    img_tensor = (img_tensor+1)/2
    
    if img_tensor.dim() == 3:
        img_tensor_mask = (img_tensor.sum(dim=0, keepdim=True)>0.001).float()
    elif img_tensor.dim() == 4:
        img_tensor_mask = (img_tensor.sum(dim=1, keepdim=True)>0.001).float()
    else:
        print("img_tensor.dim() not 3 or 4")
    
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    img_tensor_blur = gaussian_blur(img_tensor)
    img_tensor_mask_blur = gaussian_blur(img_tensor_mask)

    img_tensor_blur = img_tensor_blur/img_tensor_mask_blur
    img_tensor_blur = torch.nan_to_num(img_tensor_blur, nan=0.0)
    img_tensor_blur = img_tensor_blur*img_tensor_mask
    
    # rescale back to (-1,1)
    img_tensor_blur = img_tensor_blur*2-1

    return img_tensor_blur

def crop_n_resize(tensor, bbox, size, fill_value, mode, smooth=False):
    # input and output tensor are both with shape (n_channel,h,w)

    cv2.imwrite('__/tensor.png', ((tensor[:3,:,:].cpu().numpy().transpose((1,2,0))+1)*255/2).astype(np.uint8))

    rmin, rmax, cmin, cmax = bbox[1], bbox[3], bbox[0], bbox[2]
    tensor_crop = tensor[:, rmin:rmax+1, cmin:cmax+1]

    cv2.imwrite('__/tensor_crop.png', ((tensor_crop[:3,:,:].cpu().numpy().transpose((1,2,0))+1)*255/2).astype(np.uint8))

    # Calculate the scaling factor and resize
    crop_height, crop_width = tensor_crop.shape[1], tensor_crop.shape[2]
    scale_factor = size/max(crop_height, crop_width)

    new_height = int(crop_height*scale_factor)
    new_width = int(crop_width *scale_factor)

    if new_height == 0 or new_width == 0:
        return torch.full((tensor.shape[0], tensor.shape[1], size, size), fill_value, device=tensor.device)

    if mode == "bilinear":
        tensor_resize = F.interpolate(tensor_crop.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=True).squeeze(0)
    elif mode == "nearest-exact":
        tensor_resize = F.interpolate(tensor_crop.unsqueeze(0), size=(new_height, new_width), mode="nearest-exact").squeeze(0)

    cv2.imwrite('__/tensor_resize.png', ((tensor_resize[:3,:,:].cpu().numpy().transpose((1,2,0))+1)*255/2).astype(np.uint8))

    if smooth:
        tensor_resize = smooth_within_mask(tensor_resize, kernel_size=7, sigma=1)

    tensor_padded = torch.full((tensor_resize.shape[0], size, size), fill_value, dtype=torch.float32, device=tensor.device)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    tensor_padded[:, top:top+new_height, left:left+new_width] = tensor_resize

    cv2.imwrite('__/tensor_padded.png', ((tensor_padded[:3,:,:].cpu().numpy().transpose((1,2,0))+1)*255/2).astype(np.uint8))
    
    return tensor_padded

def create_dataloader(tensordata_filename, idx_obj_str, config, mode):

    tensordata_scene_filename = os.path.join(config["tensordata_dir"],idx_obj_str,"scene",tensordata_filename)
    tensordata_colorcode_filename = os.path.join(config["tensordata_dir"],idx_obj_str,"colorcode",tensordata_filename)

    tensordata_scene = torch.load(tensordata_scene_filename)
    tensordata_colorcode = torch.load(tensordata_colorcode_filename)

    # Define Custom Dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, tensor1, tensor2):
            self.tensor1 = tensor1
            self.tensor2 = tensor2

        def __len__(self):
            # Ensure that both tensors have the same length
            assert len(self.tensor1) == len(self.tensor2)
            return len(self.tensor1)

        def __getitem__(self, index):
            # Fetch tensors and move them to the specified device
            # return self.tensor1[index].to(self.device), self.tensor2[index].to(self.device)
            return self.tensor1[index], self.tensor2[index]

    # Create an instance of the custom dataset
    dataset = CustomDataset(tensordata_scene, tensordata_colorcode)

    if mode == "train":
        shuffle = True
    else:
        shuffle = False

    # We set drop_last to True because with ConcatDataset we want to ensure that we get 
    # one batch from dataset_scene and one batch from dataset_colorcode without leftovers.
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, drop_last=False)

    return dataloader

def find_connected_components(tensor):
    def is_valid(x, y):
        return (0 <= x < rows) and (0 <= y < cols) and tensor[x][y] == 1 and not visited[x][y]

    def bfs(x, y):
        queue = deque()
        queue.append((x, y))
        visited[x][y] = True
        component = []

        while queue:
            cx, cy = queue.popleft()
            component.append([cx, cy])

            # Check all 4 directions (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny):
                    queue.append((nx, ny))
                    visited[nx][ny] = True

        return component

    rows, cols = tensor.shape
    visited = torch.zeros_like(tensor, dtype=torch.bool)
    components = []

    for i in range(rows):
        for j in range(cols):
            if tensor[i][j] >0 and not visited[i][j]:
                component = bfs(i, j)
                components.append(component)

    return components

class Logger():
    def __init__(self, config, idx_obj_str, weight_dir):
        self.filename = os.path.join(config[weight_dir], idx_obj_str+"_log.txt")
        self.min_valid_loss =float('inf')
        self.best_model_epoch = 0

    def log(self, message, print_message=True):
        if print_message:
            print(message)
        with open(self.filename, 'a') as file:
            file.write(message + '\n')

class Dataset_colorcode(Dataset):
    def __init__(self, list_scene, list_colorcode, config, transform=base_transform, load_symm_mask=False):
        """
        Args:
            list_img_scene (list): List of paths to the scene images.
            list_img_colorcode (list): List of paths to the color mapping images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # self.list_img_scene from read txt_list_img_scene 
        self.list_img_scene = list_scene
        self.list_img_colorcode = list_colorcode
        self.transform = transform
        self.device = config["device"]
        self.load_symm_mask = load_symm_mask

    def __len__(self):
        return len(self.list_img_scene)

    def __getitem__(self, idx):
        img_scene = Image.open(self.list_img_scene[idx])
        img_colorcode = Image.open(self.list_img_colorcode[idx])

        img_scene = self.transform(img_scene)
        img_colorcode = self.transform(img_colorcode)

        if not self.load_symm_mask:
            return img_scene, img_colorcode
        else:
            symm_mask_path = (self.list_img_scene[idx]).replace('/rgb/', '/symm_mask/').replace('.jpg', '.png')
            symm_mask = Image.open(symm_mask_path)
            symm_mask = self.transform(symm_mask)
            return img_scene, img_colorcode, symm_mask

class Dataset_colorcode_with_mask(Dataset):
    def __init__(self, list_scene, list_colorcode, config, transform=base_transform, load_symm_mask=False):
        """
        Args:
            list_img_scene (list): List of paths to the scene images.
            list_img_colorcode (list): List of paths to the color mapping images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # self.list_img_scene from read txt_list_img_scene 
        self.list_img_scene = list_scene
        self.list_img_colorcode = list_colorcode
        self.transform = transform
        self.device = config["device"]
        self.num_classes = config["num_classes"]
        self.load_symm_mask = load_symm_mask

    def __len__(self):
        return len(self.list_img_scene)

    def __getitem__(self, idx):
        img_scene = Image.open(self.list_img_scene[idx])
        img_colorcode = Image.open(self.list_img_colorcode[idx])

        img_scene = self.transform(img_scene)
        img_colorcode = self.transform(img_colorcode)

        # list_obj =  [1,5,6,8,9,10,11,12]
        # idx =       [0,1,2,3,4,5, 6, 7]
        mask_gt = torch.zeros((self.num_classes, img_scene.shape[1], img_scene.shape[2]), dtype=torch.float32)
        mask_gt_obj = (torch.sum(img_colorcode, dim=0)>-0.9999*3).float()
        mask_gt_background = torch.ones_like(mask_gt_obj) - mask_gt_obj
        
        if "/000001/" in self.list_img_scene[idx]:
            mask_gt[0,:,:] = mask_gt_obj
        elif "/000005/" in self.list_img_scene[idx]:
            mask_gt[1,:,:] = mask_gt_obj
        elif "/000006/" in self.list_img_scene[idx]:
            mask_gt[2,:,:] = mask_gt_obj
        elif "/000008/" in self.list_img_scene[idx]:
            mask_gt[3,:,:] = mask_gt_obj
        elif "/000009/" in self.list_img_scene[idx]:
            mask_gt[4,:,:] = mask_gt_obj
        elif "/000010/" in self.list_img_scene[idx]:
            mask_gt[5,:,:] = mask_gt_obj
        elif "/000011/" in self.list_img_scene[idx]:
            mask_gt[6,:,:] = mask_gt_obj
        elif "/000012/" in self.list_img_scene[idx]:
            mask_gt[7,:,:] = mask_gt_obj
        mask_gt[-1,:,:] = mask_gt_background

        if not self.load_symm_mask:
            return img_scene, img_colorcode, mask_gt
        else:
            symm_mask_path = (self.list_img_scene[idx]).replace('/rgb/', '/symm_mask/').replace('.jpg', '.png')
            symm_mask = Image.open(symm_mask_path)
            symm_mask = self.transform(symm_mask)
            return img_scene, img_colorcode, mask_gt, symm_mask

class Dataset_colorcode_multi(Dataset):
    def __init__(self, list_scene, list_img, config, transform=base_transform, load_multiple=False, idx_obj=None):
        """
        Args:
            list_img_scene (list): List of paths to the scene images.
            list_img_colorcode (list): List of paths to the color mapping images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # self.list_img_scene from read txt_list_img_scene 
        self.list_img_scene = list_scene
        self.list_img = list_img
        self.transform = transform
        self.device = config["device"]
        self.load_multiple = load_multiple
        self.idx_obj = idx_obj
        self.num_classes = config["num_classes"]

    def __len__(self):
        return len(self.list_img_scene)

    def __getitem__(self, idx):
        img_scene = Image.open(self.list_img_scene[idx])
        img_scene = self.transform(img_scene)

        # get the height and width of the img_scene
        height, width = img_scene.shape[1], img_scene.shape[2]
        mask_gt = torch.zeros((self.num_classes, height, width), dtype=torch.float32)

        if self.load_multiple:
            for idx_obj in range(self.num_classes):
                mask_path = self.list_img[idx]+"_"+str(idx_obj).zfill(6)+".png"
                if os.path.exists(mask_path):
                    mask_obj = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_obj = torch.from_numpy(mask_obj).unsqueeze(0).float() / 255.0
                    mask_gt[idx_obj] = mask_obj
        else:
            # img_colorcode = Image.open(self.list_img[idx])
            img_colorcode = torch.from_numpy(cv2.imread(self.list_img[idx]))
            mask_obj = (torch.sum(img_colorcode, dim=2)>0).float()
            mask_gt[self.idx_obj] = mask_obj

        # generate the last channel as the background 
        mask_gt[-1] = 1-mask_gt.sum(dim=0, keepdim=True)

        return img_scene, mask_gt

def create_augmented_tensor_from_list(list_img_1,list_img_2, scale):

    transformed_list1 = []
    transformed_list2 = []

    deterministic_geometric_transform = DeterministicTransform(geometric_transform)

    # process original images
    for img1_path, img2_path in zip(list_img_1, list_img_2):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img1 = base_transform(img1)
        img2 = base_transform(img2)
        
        transformed_list1.append(img1)
        transformed_list2.append(img2)

    # process augmented images
    for i in range(scale-1):
        for img1_path, img2_path in zip(list_img_1, list_img_2):
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Set seed for deterministic behavior
            seed = np.random.randint(2147483647)
            deterministic_geometric_transform.set_seed(seed)

            # apply photometric transform to img1
            img1 = photometric_transfrom(img1)

            # apply the same geometric transform to both images
            img1 = deterministic_geometric_transform(img1)
            img2 = deterministic_geometric_transform(img2)

            # apply same transformation to both images
            img1 = base_transform(img1)
            img2 = base_transform(img2)
            
            transformed_list1.append(img1)
            transformed_list2.append(img2)
    
    # convert transformed_list to tensor
    datatensor_1 = torch.stack(transformed_list1)
    datatensor_2 = torch.stack(transformed_list2)
    
    return datatensor_1, datatensor_2

class ALRS:
    def __init__(self, optimizer, loss_ratio_threshold=0.01, decay_rate=0.97):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold
        self.last_loss = 999
    def step(self, loss):
        delta = self.last_loss - loss
        if delta/self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.decay_rate
        self.last_loss = loss

class Model_Noise_Injector:
    def __init__(self, std_ratio=0.01, epoch_interval=10):
        self.std_ratio = std_ratio
        self.idx_epoch_pre = 0
        self.epoch_interval = epoch_interval
        self.loss_best = np.inf
        
    def step(self,idx_epoch,model,loss_cur):
        if loss_cur <= self.loss_best:
            self.loss_best = loss_cur
            self.idx_epoch_pre = idx_epoch
        else:
            if idx_epoch-self.idx_epoch_pre >= self.epoch_interval:
                self.idx_epoch_pre = idx_epoch

                for param in model.parameters():
                    if param.requires_grad:  
                        noise = torch.randn(param.size(), device=param.device) * (self.std_ratio * param.data.std())
                        param.data += noise.to(param.device)
                print("weight noise added")

