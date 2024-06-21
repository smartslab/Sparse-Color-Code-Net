import os
import cv2
import sys
import shutil
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import colorcode_util as util
import neural_renderer as nr
from colorcode_model import UNet, initialize_weights, FocalLoss, TverskyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import concurrent.futures
from colorcode_render import xyz_euler_to_rotation_matrix

def load_sample_mask(path):
    mask_1_2 = torch.from_numpy(cv2.imread(os.path.join(path,"1_2.png"), cv2.IMREAD_GRAYSCALE)/255).to(config["device"])
    mask_1_4 = torch.from_numpy(cv2.imread(os.path.join(path,"1_4.png"), cv2.IMREAD_GRAYSCALE)/255).to(config["device"])
    mask_1_8 = torch.from_numpy(cv2.imread(os.path.join(path,"1_8.png"), cv2.IMREAD_GRAYSCALE)/255).to(config["device"])
    mask_1_9 = torch.from_numpy(cv2.imread(os.path.join(path,"1_9.png"), cv2.IMREAD_GRAYSCALE)/255).to(config["device"])

    dict_mask = {"1_2":mask_1_2, "1_4":mask_1_4, "1_8":mask_1_8, "1_9":mask_1_9}
    return dict_mask

def load_obj(filename_obj):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    textures = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            textures.append([float(v) for v in line.split()[4:7]])
    textures = torch.from_numpy(np.vstack(textures).astype(np.float32)).cuda()

    return vertices, faces, textures

def obj_loader(idx_obj_str, info_obj, models_path, colorcode_type):
    vertices, faces, textures  = nr.load_obj(models_path)

    vertices = torch.reshape(vertices, (1,vertices.shape[0],3)).cuda()
    faces = torch.reshape(faces, (1,faces.shape[0],3)).cuda()
    textures = torch.zeros(vertices.shape).cuda()
    
    # apply for rotation correction for symmetric objects
    if "symm" in colorcode_type and len(info_obj[idx_obj_str]["R_correct"])!=0:
        rot_x, rot_y, rot_z = info_obj[idx_obj_str]["R_correct"]
        R_correct = xyz_euler_to_rotation_matrix(rot_x, rot_y, rot_z, mode='deg')
        R_correct_inv = torch.tensor(R_correct, dtype=vertices.dtype, device=vertices.device).unsqueeze(0).transpose(1, 2)
        vertices = torch.bmm(vertices, R_correct_inv)

    # generate textures
    x_min = torch.min(vertices[0,:,0])
    y_min = torch.min(vertices[0,:,1])
    z_min = torch.min(vertices[0,:,2])
    x_max = torch.max(vertices[0,:,0])
    y_max = torch.max(vertices[0,:,1])
    z_max = torch.max(vertices[0,:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    r_max = max([(x_max-x_min),(y_max-y_min),(z_max-z_min)])

    if colorcode_type == "cc_origin":
        for idx_f in range(textures.shape[1]):
            textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/r_max
            textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/r_max
            textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/r_max
    elif colorcode_type == "cc_aniso":
        for idx_f in range(textures.shape[1]):
            textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/(x_max-x_min)
            textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/(y_max-y_min)
            textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/(z_max-z_min)
    elif colorcode_type == "cc_symm":
        for idx_f in range(textures.shape[1]):
            if "x" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,0] = torch.abs((vertices[0,idx_f,0]-x_mid)/(x_max-x_min)*2)
            else:
                textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/r_max
            if "y" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,1] = torch.abs((vertices[0,idx_f,1]-y_mid)/(y_max-y_min)*2)
            else:
                textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/r_max
            if "z" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,2] = torch.abs((vertices[0,idx_f,2]-z_mid)/(z_max-z_min)*2)
            else:
                textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/r_max
    elif colorcode_type == "cc_aniso_n_symm":
        for idx_f in range(textures.shape[1]):
            if "x" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,0] = torch.abs((vertices[0,idx_f,0]-x_mid)/(x_max-x_min)*2)
            else:
                textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/(x_max-x_min)
            if "y" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,1] = torch.abs((vertices[0,idx_f,1]-y_mid)/(y_max-y_min)*2)
            else:
                textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/(y_max-y_min)
            if "z" in info_obj[idx_obj_str]["symmetry"]:
                textures[0,idx_f,2] = torch.abs((vertices[0,idx_f,2]-z_mid)/(z_max-z_min)*2)
            else:
                textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/(z_max-z_min)
    elif colorcode_type == "symm_mask":
        for idx_f in range(textures.shape[1]):
            if "x" in info_obj[idx_obj_str]["symmetry"]:
                if vertices[0,idx_f,0]>0:
                    textures[0,idx_f,0] = 1
                else:
                    textures[0,idx_f,0] = 0
            else:
                textures[0,idx_f,0] = 1
            if "y" in info_obj[idx_obj_str]["symmetry"]:
                if vertices[0,idx_f,1]>0:
                    textures[0,idx_f,1] = 1
                else:
                    textures[0,idx_f,1] = 0
            else:
                textures[0,idx_f,1] = 1
            if "z" in info_obj[idx_obj_str]["symmetry"]:
                if vertices[0,idx_f,2]>0:
                    textures[0,idx_f,2] = 1
                else:
                    textures[0,idx_f,2] = 0
            else:
                textures[0,idx_f,2] = 1
    else:
        print("wrong style input")
        sys.exit()

    if colorcode_type == "symm_mask":
        return vertices, faces, [textures, torch.ones(vertices.shape).cuda()]
    else:
        return vertices, faces, textures

def inference(imgs_scene,sobel,morph_conv):
    max_num = config["num_classes"]-1
    imgs_scene = imgs_scene.to(config["device"])
    masks_contour = util.generate_mask_contour(sobel,imgs_scene)

    # Forward pass through the main network
    imgs_scene_n_contour = torch.cat((imgs_scene, masks_contour), dim=1)
    masks_raw, _ = model_seg(imgs_scene_n_contour) 
    masks_prob = F.softmax(masks_raw, dim=1)
    masks_prob_retain, scale_retain = util.retain_valid_mask(masks_prob[:,:-1,:,:], morph_conv)

    imgs_scene_n_contour_n_mask = torch.cat((imgs_scene_n_contour, masks_prob[:,:-1,:,:]), dim=1)
    crop_input = torch.zeros((max_num, imgs_scene_n_contour_n_mask.shape[1], config["square_size"], config["square_size"]), device=config["device"])
    crop_masks_contour = torch.zeros((max_num, 1, config["square_size"], config["square_size"]), device=config["device"])

    list_bbox = []
    for idx_img in range(max_num):
        bbox_pred = util.get_bbox(masks_prob_retain[0,idx_img,:,:],scale_retain)
        bbox = bbox_pred
        if bbox:
            list_bbox.append(bbox)
            crop_input[idx_img,:,:,:] = util.crop_n_resize(imgs_scene_n_contour_n_mask[0,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="bilinear", smooth=False)
            crop_masks_contour[idx_img,:,:,:] = util.crop_n_resize(masks_contour[0,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="nearest-exact", smooth=False)
    
    crop_input = mask_objchannel*crop_input
    crop_input_obj5 = crop_input[1,:,:,:]
    # repeat crop_input_obj5 8 times at the dimension 0
    crop_input = crop_input_obj5.repeat(max_num,1,1,1)
       
    crop_input = crop_input[:len(list_bbox),:,:,:]
    crop_masks_contour = crop_masks_contour[:len(list_bbox),:,:,:]

    crop_imgs_colorcode_n_mask_pred, _ = model_cc(crop_input)
    crop_imgs_colorcode_pred = torch.clamp(crop_imgs_colorcode_n_mask_pred[:,:3,:,:],min=-1,max=1)
    crop_masks_pred = (torch.sum((crop_imgs_colorcode_pred+1)/2, dim=1, keepdim=True)>0.1*3).float()
    crop_masks_contour = crop_masks_contour*crop_masks_pred
    crop_symm_masks_pred = torch.tanh(crop_imgs_colorcode_n_mask_pred[:,3:,:,:])

    # #########################################################3
    # img_scene = (imgs_scene[0,:,:,:].permute(1,2,0).cpu().numpy()+1)/2
    # mask = masks_prob[0,0,:,:].cpu().numpy()
    # img_colorcode_pred = (crop_imgs_colorcode_pred[0,:,:,:].permute(1,2,0).cpu().numpy()+1)/2
    # symm_mask_pred = (crop_symm_masks_pred[0,:,:,:].permute(1,2,0).cpu().numpy()+1)/2

    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # axs[0, 0].imshow(img_scene)
    # axs[0, 1].imshow(mask)
    # axs[1, 0].imshow(img_colorcode_pred)
    # axs[1, 1].imshow(symm_mask_pred)
    # plt.savefig("inference files/single.png")
    # plt.close()
    # ###########################################################
    
    list_pix2D = []
    list_pix3D = []
    crop_imgs_colorcode_pred = (crop_imgs_colorcode_pred[:,:,:,:]+1)/2
    crop_symm_masks_pred = crop_symm_masks_pred*crop_masks_contour

    for idx_img in range(max_num):
        if idx_img == 0:
            idx_obj_str = "000001"
        elif idx_img == 1:
            idx_obj_str = "000005"
        elif idx_img == 2:
            idx_obj_str = "000006"
        elif idx_img == 3:
            idx_obj_str = "000008"
        elif idx_img == 4:
            idx_obj_str = "000009"
        elif idx_img == 5:
            idx_obj_str = "000010"
        elif idx_img == 6:
            idx_obj_str = "000011"
        elif idx_img == 7:
            idx_obj_str = "000012"

        bbox = list_bbox[0]
        rmin, rmax, cmin, cmax = bbox[1], bbox[3], bbox[0], bbox[2]
        crop_height = rmax-rmin+1
        crop_width = cmax-cmin+1
        scale_factor = config["square_size"]/max(crop_height, crop_width)
        
        img_colorcode_pred = crop_imgs_colorcode_pred[idx_img,:,:,:]
        symm_mask_pred = crop_symm_masks_pred[0,:,:,:]*crop_masks_contour[0,0:1,:,:]
        num_symm = torch.sum(torch.abs(symm_mask_pred)>(0.7*3)).item()
        if num_symm<200:
            symm_mask_pred = symm_mask_pred*dict_mask["1_2"]
        elif num_symm<400:
            symm_mask_pred = symm_mask_pred*dict_mask["1_4"]
        elif num_symm<800:
            symm_mask_pred = symm_mask_pred*dict_mask["1_8"]
        else:
            symm_mask_pred = symm_mask_pred*dict_mask["1_9"]

        # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # axs[0, 0].imshow(mask_pos.cpu().numpy())
        # axs[0, 1].imshow(mask_neg.cpu().numpy())
        # plt.savefig("inference files/single.png")
        # plt.close()

        if "symm" in colorcode_type and len(info_obj[idx_obj_str]["R_correct"])!=0:
            if "x" in info_obj[idx_obj_str]["symmetry"]:
                mask_pos = (symm_mask_pred[0,:,:]>0.7).int()
                mask_neg = (symm_mask_pred[0,:,:]<-0.7).int()
            elif "y" in info_obj[idx_obj_str]["symmetry"]:
                mask_pos = (symm_mask_pred[1,:,:]>0.7).int()
                mask_neg = (symm_mask_pred[1,:,:]<-0.7).int()

            idx_pos = mask_pos.nonzero()
            idx_neg = mask_neg.nonzero()
            pix_cc_pos = img_colorcode_pred[:, idx_pos[:, 0], idx_pos[:, 1]].t()
            pix_cc_neg = img_colorcode_pred[:, idx_neg[:, 0], idx_neg[:, 1]].t()

            if "x" in info_obj[idx_obj_str]["symmetry"]:
                pix_cc_neg[0,:] = -pix_cc_neg[0,:]
            if "y" in info_obj[idx_obj_str]["symmetry"]:
                pix_cc_neg[1,:] = -pix_cc_neg[1,:]

            pix_2D_pos = torch.cat((idx_neg[:,1]/scale_factor+cmin, idx_neg[:,0]/scale_factor+rmin), dim=1).to(torch.float64)
            pix_2D_neg = torch.cat((idx_pos[:,1]/scale_factor+cmin, idx_pos[:,0]/scale_factor+rmin), dim=1).to(torch.float64)

            pix_2D = torch.cat((pix_2D_pos, pix_2D_neg), dim=0).to(torch.float64).cpu().numpy()
            pix_cc = torch.cat((pix_cc_pos, pix_cc_neg), dim=0).to(torch.float64).cpu().numpy()
        else:
            mask_pos = (symm_mask_pred[0,:,:]>0.7).int()
            idx_pos = mask_pos.nonzero()
            pix_cc_pos = img_colorcode_pred[:, idx_pos[:, 0], idx_pos[:, 1]].t()
            pix_2D_pos = torch.cat((idx_pos[:,1]/scale_factor+cmin, idx_pos[:,0]/scale_factor+rmin), dim=1).to(torch.float64)
            pix_2D = pix_2D_pos.to(torch.float64).cpu().numpy()
            pix_cc = pix_cc_pos.to(torch.float64).cpu().numpy

        R_pred,t_pred = PnP(pix_cc, pix_2D, vertices, idx_obj_str, info_obj, cam_K, colorcode_type)

    for i in range(len(list_pix2D)):
        pix_3D = list_pix3D[i]
        pix_2D = list_pix2D[i]
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(pix_3D, pix_2D, cam_K,None,reprojectionError=5,iterationsCount=100)
        R_pred = np.eye(3)
        t_pred = tvec[:,0].reshape(3,1)
        cv2.Rodrigues(rvec, R_pred) 

def PnP(pix_cc, pix_2D, vertices, idx_obj_str, info_obj, cam_K, colorcode_type):

    # apply for rotation correction for symmetric objects
    if "symm" in colorcode_type and len(info_obj[idx_obj_str]["R_correct"])!=0:
        rot_x, rot_y, rot_z = info_obj[idx_obj_str]["R_correct"]
        R_correct = xyz_euler_to_rotation_matrix(rot_x, rot_y, rot_z, mode='deg')

    x_min = torch.min(vertices[0,:,0]).item()
    y_min = torch.min(vertices[0,:,1]).item()
    z_min = torch.min(vertices[0,:,2]).item()
    x_max = torch.max(vertices[0,:,0]).item()
    y_max = torch.max(vertices[0,:,1]).item()
    z_max = torch.max(vertices[0,:,2]).item()
    r_max = max([(x_max-x_min),(y_max-y_min),(z_max-z_min)])

    if colorcode_type == "cc_aniso" or colorcode_type == "cc_aniso_n_symm":
        pix_3D = np.vstack((pix_cc[:,0]*(x_max-x_min)+x_min,pix_cc[:,1]*(y_max-y_min)+y_min,pix_cc[:,2]*(z_max-z_min)+z_min)).T
    elif colorcode_type == "cc_origin":
        pix_3D = np.vstack((pix_cc[:,0]*r_max+x_min,pix_cc[:,1]*r_max+y_min,pix_cc[:,2]*r_max+z_min)).T

    # recover the point cloud translated from colorcode to the original dataset position
    if len(info_obj[idx_obj_str]["R_correct"])!=0:
        pix_3D = np.dot(pix_3D, R_correct)  

    # if num_sample!=0:
    #     idx_select = random.sample(range(len(pix_2D)), num_sample)
    #     pix_3D = pix_3D[idx_select,:]
    #     pix_2D = pix_2D[idx_select,:]

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(pix_3D, pix_2D, cam_K,None,reprojectionError=5,iterationsCount=100)

    R_pred = np.eye(3)
    t_pred = tvec[:,0].reshape(3,1)
    cv2.Rodrigues(rvec, R_pred)  
    return R_pred,t_pred


if __name__ == '__main__':
    # load configuration
    config_file = "config_multi.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    with open("config_obj.yaml", "r") as file:
        info_obj = yaml.safe_load(file)
    dict_mask = load_sample_mask("sample_mask")
    
    colorcode_type = "cc_aniso_n_symm"
    idx_obj_str = "000001"
    models_folder = "/content/drive/MyDrive/data/LINEMOD/models/"
    cam_K = np.array([[572.4114, 0.0, 325.2611],[0.0, 573.57043, 242.04899],[0,0,1]]).astype(np.float64)
    models_path = models_folder+"obj_"+idx_obj_str+".obj"
    vertices, faces, textures = obj_loader(idx_obj_str, info_obj, models_path, colorcode_type)

    # create sobel filter
    sobel = util.sobel_generator().to(config["device"])
    morph_conv = torch.nn.Conv2d(config["num_classes"]-1, config["num_classes"]-1, kernel_size=3, stride=1, padding=1, groups=config["num_classes"]-1).to(config["device"])
    morph_conv.weight.data.fill_(1.0)

    mask_objchannel = torch.zeros((8,4+config["num_classes"]-1,128,128)).to(config["device"])
    mask_objchannel[:,:4,:,:] = 1
    for i in range(config["num_classes"]-1):
        mask_objchannel[i,4+i,:,:] = 1

    # initialize the network and optimizer
    model_seg = UNet(num_depth=config["num_depth_scene"], 
                    num_basefilter=config["num_basefilter_scene"], 
                    input_channels=config["input_channels_scene"], 
                    output_channels=config["num_classes"],
                    kernel_size_down = config["kernel_size_scened_down"],
                    kernel_size_up = config["kernel_size_scened_up"],
                    ).to(config["device"])
    model_cc = UNet(num_depth=config["num_depth_object"], 
                    num_basefilter=config["num_basefilter_object"], 
                    input_channels=4+config["num_classes"]-1, #  RGB + contour + mask_o/_background
                    output_channels=config["output_channels_object"],
                    kernel_size_down = config["kernel_size_objectd_down"],
                    kernel_size_up = config["kernel_size_objectd_up"],
                    ).to(config["device"])  
    
    model_seg.load_state_dict(torch.load("inference files/seg_multi.pth"))
    model_cc.load_state_dict(torch.load("inference files/cc_multi.pth"))
    model_seg.eval()
    model_cc.eval()
    torch.set_grad_enabled(False)

    imgs_scene = cv2.imread("/content/drive/MyDrive/colorcode_estimator/inference files/test1.jpg")
    imgs_scene = cv2.cvtColor(imgs_scene, cv2.COLOR_BGR2RGB)
    imgs_scene = imgs_scene.astype(np.float32)/255.0*2-1
    imgs_scene = torch.from_numpy(imgs_scene).permute(2,0,1).unsqueeze(0)

    for i in range(5):
        inference(imgs_scene,sobel,morph_conv)

    torch.cuda.synchronize()  # Wait for all current GPU tasks to complete
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    for i in range(50):
        inference(imgs_scene,sobel,morph_conv)

    end_time.record()
    torch.cuda.synchronize()  

    elapsed_time_ms = start_time.elapsed_time(end_time)/50 
    print(f"Operation time: {elapsed_time_ms / 1000} seconds")




