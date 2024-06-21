import os
import cv2
import shutil
import random
import yaml
import torch
import torch.nn as nn
import colorcode_util as util
from colorcode_model import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import gc

# todo (for output, 0 is grey, -1 is black)
def run_epoch(model_seg, mode, train_section, dataloader, optimizer, config, idx_epoch, logger):

    if mode == "train":
        model_seg.train()
        torch.set_grad_enabled(True)
    elif mode == "valid" or mode == "record":
        model_seg.eval()
        torch.set_grad_enabled(False)
        
    counter = 0
    counter_invalid_iou = 0
    counter_invalid_0iou = 0
    
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

        for idx_img in range(imgs_scene.shape[0]):
            idx_img_str = list_scene_valid[counter+idx_img][-10:-4]

            bbox_gt = util.get_bbox(masks_gt_obj[idx_img,0,:,:])
            bbox_pred = util.get_bbox(masks_prob_retain[idx_img,0,:,:],scale_retain)
            bbox = bbox_pred

            if bbox_pred is None:
                iou = 0
            else:
                s_gt = (bbox_gt[2]-bbox_gt[0])*(bbox_gt[3]-bbox_gt[1])
                s_pred = (bbox_pred[2]-bbox_pred[0])*(bbox_pred[3]-bbox_pred[1])
                s_intersect = (min(bbox_gt[2],bbox_pred[2])-max(bbox_gt[0],bbox_pred[0]))*(min(bbox_gt[3],bbox_pred[3])-max(bbox_gt[1],bbox_pred[1]))
                s_union = s_gt+s_pred-s_intersect
                iou = s_intersect/s_union
            if iou<0.5:
                if iou==0:
                    counter_invalid_0iou += 1
                counter_invalid_iou += 1
                    
                h,w = masks_gt[idx_img,0,:,:].shape

                k1, k2, k3 = 16, 8, 8

                masks_pred_h = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k1, stride=k1)>0.9).float()[:,:-1,:,:]
                masks_pred_m = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k2, stride=k2)>0.7).float()[:,:-1,:,:]
                masks_pred_l = (torch.nn.functional.max_pool2d(masks_prob, kernel_size=k3, stride=k3)>0.5).float()[:,:-1,:,:]

                masks_pred_h1 = torch.nn.functional.interpolate(masks_pred_h, scale_factor=k1)
                masks_pred_h = (morph_conv(masks_pred_h)>0.5).float()
                masks_pred_h2 = torch.nn.functional.interpolate(masks_pred_h, scale_factor=k1)
                masks_pred_h = torch.nn.functional.interpolate(masks_pred_h, scale_factor=int(k1/k2))
                masks_pred_m1 = torch.nn.functional.interpolate(masks_pred_m, scale_factor=k2)
                masks_pred_m = masks_pred_m*masks_pred_h
                masks_pred_m2 = torch.nn.functional.interpolate(masks_pred_m, scale_factor=k2)

                masks_pred_m = (morph_conv(masks_pred_m)>0.5).float()
                masks_pred_m3 = torch.nn.functional.interpolate(masks_pred_m, scale_factor=k2)
                masks_pred_m = torch.nn.functional.interpolate(masks_pred_m, scale_factor=int(k2/k3))
                masks_pred_l1 = torch.nn.functional.interpolate(masks_pred_l, scale_factor=k3)
                masks_pred_l = masks_pred_l*masks_pred_m
                masks_pred_l2 = torch.nn.functional.interpolate(masks_pred_l, scale_factor=k3)
                masks_pred_l = (morph_conv(masks_pred_l)>0.5).float()
                masks_pred_l3 = torch.nn.functional.interpolate(masks_pred_l, scale_factor=k3)
                
                masks_pred_final = masks_pred_l2
                masks_pred = (masks_prob>0.5).float()

                mask_pred_h = -torch.ones((3,h,w))
                mask_pred_h[0,:,:] = (masks_pred[idx_img,0,:,:]-0.5)*2
                mask_pred_h[1,:,:] = ((masks_pred_h1[idx_img,0,:,:]-0.5)*2+(masks_pred_h2[idx_img,0,:,:]-0.5)*2)/2

                mask_pred_m = -torch.ones((3,h,w))
                mask_pred_m[0,:,:] = (masks_pred[idx_img,0,:,:]-0.5)*2
                mask_pred_m[1,:,:] = ((masks_pred_m1[idx_img,0,:,:]-0.5)*2+(masks_pred_m2[idx_img,0,:,:]-0.5)*2+(masks_pred_m3[idx_img,0,:,:]-0.5)*2)/3

                mask_pred_l = -torch.ones((3,h,w))
                mask_pred_l[0,:,:] = (masks_pred[idx_img,0,:,:]-0.5)*2
                mask_pred_l[1,:,:] = ((masks_pred_l1[idx_img,0,:,:]-0.5)*2+(masks_pred_l2[idx_img,0,:,:]-0.5)*2+(masks_pred_l3[idx_img,0,:,:]-0.5)*2)/3

                mask_pred_n_gt = -torch.ones((3,h,w))
                mask_pred_n_gt[0,:,:] = (masks_pred_final[idx_img,0,:,:]-0.5)*2
                mask_pred_n_gt[1,:,:] = (masks_gt[idx_img,0,:,:]-0.5)*2

                util.save_image_from_tensor(masks_prob[idx_img,0,:,:], os.path.join(config["record_dir"],train_section,idx_obj_str,idx_img_str+"_mask_prob.png"))
                util.save_image_from_tensor(mask_pred_h, os.path.join(config["record_dir"],train_section,idx_obj_str,idx_img_str+"_mask_pred1.png"))
                util.save_image_from_tensor(mask_pred_m, os.path.join(config["record_dir"],train_section,idx_obj_str,idx_img_str+"_mask_pred2.png"))
                util.save_image_from_tensor(mask_pred_l, os.path.join(config["record_dir"],train_section,idx_obj_str,idx_img_str+"_mask_pred3.png"))
                util.save_image_from_tensor(mask_pred_n_gt, os.path.join(config["record_dir"],train_section,idx_obj_str,idx_img_str+"_mask_pred_n_gt.png"))

        counter = counter+masks_raw.shape[0]
    print("obj",idx_obj_str,"counter_invalid_iou: ", counter_invalid_iou, counter_invalid_0iou)

    return 0

# load configuration
if __name__ == '__main__':
    # load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    with open("config_obj.yaml", "r") as file:
        info_obj = yaml.safe_load(file)

    train_section = "segmentation"
    weight_dir = "weight_"+train_section+"_dir"
    colorcode_type = "cc_aniso_n_symm"

    # create sobel filter
    sobel = util.sobel_generator().to(config["device"])
    morph_conv = torch.nn.Conv2d(config["num_classes"]-1, config["num_classes"]-1, kernel_size=3, stride=1, padding=1, groups=config["num_classes"]-1).to(config["device"])
    morph_conv.weight.data.fill_(1.0)

    for idx_obj in [1,2,4,5,6,8,9,10,11,12,13,14,15]:
    
        idx_obj_str = str(idx_obj).zfill(6)
        if os.path.exists(os.path.join(config["record_dir"],train_section,idx_obj_str)):
            shutil.rmtree(os.path.join(config["record_dir"],train_section,idx_obj_str))
        os.makedirs(os.path.join(config["record_dir"],train_section,idx_obj_str))

        # get the list of images
        with open(os.path.join("data list",idx_obj_str,"scene_valid.txt"), 'r') as file:
            list_scene_valid = [line.rstrip() for line in file.readlines()]
        with open(os.path.join("data list",idx_obj_str,"colorcode_valid.txt"), 'r') as file:
            list_colorcode_valid = [line.rstrip() for line in file.readlines()]

        # create dataloader
        mp.set_start_method('spawn', force=True)
        # create dataloader
        mp.set_start_method('spawn', force=True)
        if "symm" in colorcode_type:
            load_symm_mask = True
        else:
            load_symm_mask = False
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
        
        # load best model and save the result visualization
        idx_epoch = 0
        model_seg.load_state_dict(torch.load(os.path.join(config[weight_dir],idx_obj_str+"_best_model.pth")))
        optimizer = torch.optim.Adam(model_seg.parameters(), lr=config["learning_rate"])
        _ = run_epoch(model_seg, "record", train_section, dataloader_valid, optimizer, config, idx_epoch, logger)
        print("idx_obj_str: ", idx_obj_str, " completed")







