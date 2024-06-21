import os
import cv2
import shutil
import yaml
import torch
import random
import json
import math
import numpy as np
import torch.nn as nn
import colorcode_util as util
from colorcode_model import UNet, initialize_weights, FocalLoss, TverskyLoss
from colorcode_render import obj_loader, PnP, xyz_euler_to_rotation_matrix, render_colorcode
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# todo (for output, 0 is grey, -1 is black)
def run_epoch(model_seg, model_cc, mode, train_section, dataloader, optimizer, config, idx_epoch, logger):

    if mode == "train":
        model_cc.train()
        torch.set_grad_enabled(True)
    elif mode == "valid" or mode == "record":
        model_cc.eval()
        torch.set_grad_enabled(False)
        
    counter = 0
    
    for idx_batch, data in enumerate(dataloader):
        if "symm" in colorcode_type:
            imgs_scene, imgs_colorcode_gt, symm_masks = data
        else:
            imgs_scene, imgs_colorcode_gt = data

        batch_size = imgs_scene.shape[0]
        imgs_scene = imgs_scene.to(config["device"])
        imgs_colorcode_gt = imgs_colorcode_gt.to(config["device"])
        
        # get the mask from the imgs_colorcode_gt, stored as channels, last is background  todo multiple classes, last being backgrounds?
        masks_gt_obj = (torch.sum(imgs_colorcode_gt, dim=1, keepdim=True)>-0.9999*3).float()
        masks_gt_background = torch.ones_like(masks_gt_obj) - masks_gt_obj
        masks_gt = torch.cat((masks_gt_obj, masks_gt_background), dim=1)
        masks_contour = util.generate_mask_contour(sobel,imgs_scene)
        
        # Forward pass through the main network
        imgs_scene_n_contour = torch.cat((imgs_scene, masks_contour), dim=1)
        masks_raw, _ = model_seg(imgs_scene_n_contour) 
        masks_prob = F.softmax(masks_raw, dim=1)
        masks_prob_retain, scale_retain = util.retain_valid_mask(masks_prob[:,:-1,:,:], morph_conv)
        
        imgs_scene_n_contour_n_mask = torch.cat((imgs_scene_n_contour, masks_prob[:,:-1,:,:]), dim=1)
        crop_input = torch.zeros((batch_size, imgs_scene_n_contour_n_mask.shape[1], config["square_size"], config["square_size"]), device=config["device"])
        crop_imgs_colorcode_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])
        crop_masks_contour = torch.zeros((batch_size, 1, config["square_size"], config["square_size"]), device=config["device"])
        if "symm" in colorcode_type:
            crop_symm_masks_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])

        list_bbox = []
        list_bbox_gt = []
        list_bbox_pred = []
        for idx_img in range(imgs_scene.shape[0]):
            bbox_gt = util.get_bbox(masks_gt_obj[idx_img,0,:,:])
            bbox_pred = util.get_bbox(masks_prob_retain[idx_img,0,:,:],scale_retain)
            bbox = bbox_pred
            list_bbox_gt.append(bbox_gt)
            list_bbox_pred.append(bbox_pred)
            list_bbox.append(bbox)

            crop_input[idx_img,:,:,:] = util.crop_n_resize(imgs_scene_n_contour_n_mask[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="bilinear", smooth=False)
            crop_imgs_colorcode_gt[idx_img,:,:,:] = util.crop_n_resize(imgs_colorcode_gt[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=-1, mode="nearest-exact", smooth=True)
            crop_masks_contour[idx_img,:,:,:] = util.crop_n_resize(masks_contour[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="nearest-exact", smooth=False)
            if "symm" in colorcode_type:
                crop_symm_masks_gt[idx_img,:,:,:] = util.crop_n_resize(symm_masks[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="nearest-exact", smooth=False)

        crop_imgs_colorcode_n_mask_pred, _ = model_cc(crop_input)
        crop_imgs_colorcode_pred = torch.clamp(crop_imgs_colorcode_n_mask_pred[:,:3,:,:],min=-1,max=1)
        crop_masks_gt = (torch.sum(crop_imgs_colorcode_gt, dim=1, keepdim=True)>-0.5*3).float()
        # crop_masks_pred = (torch.sum(crop_imgs_colorcode_pred, dim=1, keepdim=True)>-0.2*3).float()
        crop_masks_pred = (torch.sum((crop_imgs_colorcode_pred+1)/2, dim=1, keepdim=True)>0.1*3).float()
        crop_masks_contour = crop_masks_contour*crop_masks_gt
        if "symm" in colorcode_type:
            crop_symm_masks_pred = crop_imgs_colorcode_n_mask_pred[:,3:,:,:]
            crop_symm_masks_pred = torch.tanh(crop_symm_masks_pred)

        # convert the tensor from range [-1,1] to [0,255]
        imgs_scene = (imgs_scene+1)/2*255
        imgs_colorcode_gt = (imgs_colorcode_gt+1)/2*255
        symm_masks = (symm_masks+1)/2*255
        crop_imgs_colorcode_gt = (crop_imgs_colorcode_gt+1)/2*255
        crop_imgs_colorcode_pred = (crop_imgs_colorcode_pred+1)/2*255
        crop_masks_gt = crop_masks_gt*255
        crop_masks_pred = crop_masks_pred*255
        crop_masks_contour = crop_masks_contour*255
        if "symm" in colorcode_type:
            crop_symm_masks_gt = (crop_symm_masks_gt+1)/2*255
            crop_symm_masks_pred = (crop_symm_masks_pred+1)/2*255

        for idx_img in range(imgs_scene.shape[0]):
            bbox = list_bbox[idx_img]
            bbox_gt = list_bbox_gt[idx_img]
            bbox_pred = list_bbox_pred[idx_img]
            idx_img_valid = int(list_scene_valid[counter+idx_img][-10:-4])
            cmin, rmin, cmax, rmax = list_bbox[idx_img]
            
            img_cc_path = os.path.join("/content/LINEMOD/base", idx_obj_str, colorcode_type, str(idx_img_valid).zfill(6)+".png")
            img_scene_path = os.path.join("/content/LINEMOD/base", idx_obj_str, "rgb", str(idx_img_valid).zfill(6)+".jpg")
            
            list_obj_pose = scene_gt_data[str(idx_img_valid)]
            cam_K = np.array(scene_camera_data[str(idx_img_valid)]['cam_K']).reshape((3,3)).astype(np.float32)
            R_gt = np.array(list_obj_pose[0]["cam_R_m2c"]).reshape((3,3)).astype(np.float32)
            t_gt = np.array(list_obj_pose[0]["cam_t_m2c"]).reshape((3,1)).astype(np.float32)
            print("t_gt",t_gt.T)

            img_cc_gt = crop_imgs_colorcode_gt[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()
            mask = crop_masks_gt[idx_img,0,:,:].cpu().numpy()
            R_cc_gt,t_cc_gt = PnP(img_cc_gt, mask, vertices, idx_obj_str, info_obj, cam_K, colorcode_type, bbox)
            print("t_cc_gt",t_cc_gt.T)

            img_cc_pred = (crop_imgs_colorcode_pred*crop_masks_contour/255)[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()
            mask = crop_masks_contour[idx_img,0,:,:].cpu().numpy()
            print("img_cc_pred min max, mask min max",np.min(img_cc_pred),np.max(img_cc_pred),np.min(mask),np.max(mask))
            R_cc_pred,t_cc_pred = PnP(img_cc_pred, mask, vertices, idx_obj_str, info_obj, cam_K, colorcode_type, bbox)
            print("t_cc_contour_pred",t_cc_pred.T)
            
            img_cc_pred = crop_imgs_colorcode_pred[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()
            # mask = crop_masks_pred[0,:,:].cpu().numpy()
            mask = crop_masks_gt[idx_img,0,:,:].cpu().numpy()
            print("img_cc_pred min max, mask min max",np.min(img_cc_pred),np.max(img_cc_pred),np.min(mask),np.max(mask))
            R_cc_pred,t_cc_pred = PnP(img_cc_pred, mask, vertices, idx_obj_str, info_obj, cam_K, colorcode_type, bbox)
            print("t_cc_pred",t_cc_pred.T)

            img_scene_cc_gt = imgs_colorcode_gt[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()
            mask = masks_gt_obj[idx_img,0,:,:].cpu().numpy()
            R_img_gt,t_img_gt = PnP(img_scene_cc_gt, mask, vertices, idx_obj_str, info_obj, cam_K, colorcode_type)
            print("t_img_gt",t_img_gt.T)

            if True: # plot
                img_cc_ref = cv2.imread(img_cc_path)
                img_cc_ref = cv2.cvtColor(img_cc_ref, cv2.COLOR_BGR2RGB)
                img_scene_ref = cv2.imread(img_scene_path)
                img_scene_ref = cv2.cvtColor(img_scene_ref, cv2.COLOR_BGR2RGB)
                img_cc_gt = imgs_colorcode_gt[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()
                img_scene_gt = imgs_scene[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy()

                fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                axs[0, 0].imshow(img_cc_ref)
                axs[0, 1].imshow(img_cc_gt.astype(np.uint8))
                axs[1, 0].imshow(img_scene_ref)
                axs[1, 1].imshow(img_scene_gt.astype(np.uint8))
                axs[0, 0].set_title('img_cc_ref')
                axs[0, 1].set_title('img_cc_gt')
                axs[1, 0].set_title('img_scene_ref')
                axs[1, 1].set_title('img_scene_gt')
                rect_gt = Rectangle((bbox_gt[0], bbox_gt[1]), bbox_gt[2]-bbox_gt[0], bbox_gt[3]-bbox_gt[1], linewidth=2, edgecolor='g', facecolor='none')
                rect_pred = Rectangle((bbox_pred[0], bbox_pred[1]), bbox_pred[2]-bbox_pred[0], bbox_pred[3]-bbox_pred[1], linewidth=2, edgecolor='b', facecolor='none')
                axs[1, 0].add_patch(rect_gt)
                axs[1, 0].add_patch(rect_pred)
                plt.tight_layout()
                plt.savefig("_/plot_scene_n_cc.png")
                plt.close()

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                axs[0, 0].imshow(crop_masks_gt[idx_img,0,:,:].cpu().numpy(), cmap='gray')
                axs[0, 1].imshow(crop_masks_pred[idx_img,0,:,:].cpu().numpy(), cmap='gray')
                axs[1, 0].imshow(crop_masks_contour[idx_img,0,:,:].cpu().numpy(), cmap='gray')
                axs[0, 0].set_title('mask_gt')
                axs[0, 1].set_title('mask_pred')
                axs[1, 0].set_title('mask_contour')
                plt.tight_layout()
                plt.savefig("_/plot_mask.png")
                plt.close()

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                axs[0,0].imshow(crop_imgs_colorcode_gt[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                axs[0,1].imshow(crop_imgs_colorcode_pred[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                axs[0,2].imshow((crop_imgs_colorcode_pred*crop_masks_contour/255)[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                axs[1,0].imshow(crop_symm_masks_gt[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                axs[1,1].imshow(crop_symm_masks_pred[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                axs[0,0].set_title('img_cc_gt')
                axs[0,1].set_title('img_cc_pred')
                axs[0,2].set_title('img_cc_pred*countour')
                axs[1,0].set_title('symm_mask_gt')
                axs[1,1].set_title('symm_mask_pred')
                plt.tight_layout()
                plt.savefig("_/plot_cc_crop.png")
                plt.close()
                
                bgr_image = cv2.cvtColor(crop_imgs_colorcode_pred[idx_img,:,:,:].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                cv2.imwrite('_/img_cc_pred.png', bgr_image)

            _ = 0

        counter = counter+batch_size


if __name__ == '__main__':
    # load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    with open("config_obj.yaml", "r") as file:
        info_obj = yaml.safe_load(file)

    colorcode_type = "cc_aniso_n_symm"
    train_section = "colorcode"
    weight_dir = "weight_"+train_section+"_dir" 

    # create sobel filter and morphological convolution
    sobel = util.sobel_generator().to(config["device"])
    morph_conv = torch.nn.Conv2d(config["num_classes"]-1, config["num_classes"]-1, kernel_size=3, stride=1, padding=1, groups=config["num_classes"]-1).to(config["device"])
    morph_conv.weight.data.fill_(1.0)

    for idx_obj in [1,2,4,5,6,8,9,10,11,12,13,14,15]:
        idx_obj_str = str(idx_obj).zfill(6)

        models_folder = "/content/drive/MyDrive/data/LINEMOD/models/"
        models_path = models_folder+"obj_"+idx_obj_str+".obj"
        colorcode_type = "cc_aniso_n_symm"
        with open(os.path.join("/content/LINEMOD/base", idx_obj_str, "scene_gt.json")) as f:
            scene_gt_data = json.load(f)
        with open(os.path.join("/content/LINEMOD/base", idx_obj_str, "scene_camera.json")) as f:
            scene_camera_data = json.load(f)

        vertices, faces, textures = obj_loader(idx_obj_str, info_obj, models_path, colorcode_type)

        # get the list of images
        with open(os.path.join("data list",idx_obj_str,"scene_valid.txt"), 'r') as file:
            list_scene_valid = [line.rstrip() for line in file.readlines()]
        with open(os.path.join("data list",idx_obj_str,"colorcode_valid.txt"), 'r') as file:
            list_colorcode_valid = [line.rstrip() for line in file.readlines()]

        # create dataloader
        if "symm" in colorcode_type:
            load_symm_mask = True
        else:
            load_symm_mask = False
        mp.set_start_method('spawn', force=True)
        dataloader_valid = DataLoader(util.Dataset_colorcode(list_scene_valid, list_colorcode_valid, config, load_symm_mask=load_symm_mask), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=False)

        # create logger
        logger = util.Logger(config, idx_obj_str, weight_dir)

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
        model_seg.load_state_dict(torch.load(os.path.join(config["weight_segmentation_dir"],idx_obj_str+"_best_model.pth")))
        model_cc.load_state_dict(torch.load(os.path.join(config["weight_colorcode_dir"],idx_obj_str+"_best_model.pth")))
        
        optimizer = torch.optim.Adam(model_cc.parameters(), lr=config["learning_rate"])
            
        # train and valid the network
        run_epoch(model_seg, model_cc,  "valid", train_section, dataloader_valid, optimizer, config, 0, logger)





