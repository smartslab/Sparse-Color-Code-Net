import os
import cv2
import shutil
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import colorcode_util as util
from colorcode_model import UNet, initialize_weights, FocalLoss, TverskyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

def run_epoch(model_seg, model_cc, mode, train_section, dataloader, optimizer, config, idx_epoch, logger):

    if mode == "train":
        model_cc.train()
        torch.set_grad_enabled(True)
    elif mode == "valid" or mode == "record":
        model_cc.eval()
        torch.set_grad_enabled(False)
        
    counter = 0
    idx_batch = 0
    sum_loss_cc1 = 0
    sum_loss_cc2 = 0
    sum_loss_perpixel = 0
    sum_loss_perpixel_contour = 0

    time_start = time.time()
    
    for idx_batch, data in enumerate(dataloader):
        if "symm" in colorcode_type:
            imgs_scene, imgs_colorcode_gt, symm_masks = data
        else:
            imgs_scene, imgs_colorcode_gt = data

        optimizer.zero_grad()

        batch_size = imgs_scene.shape[0]
        imgs_scene = imgs_scene.to(config["device"])
        imgs_colorcode_gt = imgs_colorcode_gt.to(config["device"])
        if "symm" in colorcode_type:
            symm_masks = symm_masks.to(config["device"])
        
        # get the sparse contour of the image todo
        masks_contour = util.generate_mask_contour(sobel,imgs_scene)

        # get the mask from the imgs_colorcode_gt, stored as channels, last is background  todo multiple classes, last being backgrounds?
        masks_gt_obj = (torch.sum(imgs_colorcode_gt, dim=1, keepdim=True)>-0.9999*3).float()
        masks_gt_background = torch.ones_like(masks_gt_obj) - masks_gt_obj
        masks_gt = torch.cat((masks_gt_obj, masks_gt_background), dim=1)
        
        # Forward pass through the main network
        imgs_scene_n_contour = torch.cat((imgs_scene, masks_contour), dim=1)
        masks_raw, _ = model_seg(imgs_scene_n_contour) 
        masks_prob = F.softmax(masks_raw, dim=1)
        imgs_scene_n_contour_n_mask = torch.cat((imgs_scene_n_contour, masks_prob[:,:-1,:,:]), dim=1)
        
        crop_input = torch.zeros((batch_size, imgs_scene_n_contour_n_mask.shape[1], config["square_size"], config["square_size"]), device=config["device"])
        crop_imgs_colorcode_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])
        crop_masks_contour = torch.zeros((batch_size, 1, config["square_size"], config["square_size"]), device=config["device"])
        if "symm" in colorcode_type:
            crop_symm_masks_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])

        for idx_img in range(imgs_scene.shape[0]):

            bbox_gt = util.get_bbox(masks_gt_obj[idx_img,0,:,:])
            if bbox_gt is None:
                continue
            h_bbox, w_bbox = bbox_gt[3]-bbox_gt[1]+1, bbox_gt[2]-bbox_gt[0]+1
            h_img,w_img = masks_gt_obj.shape[2], masks_gt_obj.shape[3]

            # with 20% variation random bbox
            random_rmin = int(((random.random()-0.5)*0.4*h_bbox)+bbox_gt[1])
            random_rmax = int(((random.random()-0.5)*0.4*h_bbox)+bbox_gt[3])
            random_cmin = int(((random.random()-0.5)*0.4*w_bbox)+bbox_gt[0])
            random_cmax = int(((random.random()-0.5)*0.4*w_bbox)+bbox_gt[2])

            random_rmin = max(0, random_rmin)
            random_rmax = min(h_img-1, random_rmax)
            random_cmin = max(0, random_cmin)
            random_cmax = min(w_img-1, random_cmax)
            bbox = [random_cmin, random_rmin, random_cmax, random_rmax]

            crop_input[idx_img,:,:,:] = util.crop_n_resize(imgs_scene_n_contour_n_mask[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="bilinear", smooth=False)
            crop_imgs_colorcode_gt[idx_img,:,:,:] = util.crop_n_resize(imgs_colorcode_gt[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=-1, mode="nearest-exact", smooth=True)
            crop_masks_contour[idx_img,:,:,:] = util.crop_n_resize(masks_contour[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="nearest-exact", smooth=False)
            if "symm" in colorcode_type:
                crop_symm_masks_gt[idx_img,:,:,:] = util.crop_n_resize(symm_masks[idx_img,:,:,:], bbox, size=config["square_size"], fill_value=0, mode="nearest-exact", smooth=False)

        crop_img_colorcode_n_mask_pred, _ = model_cc(crop_input)
        crop_img_colorcode_pred = crop_img_colorcode_n_mask_pred[:,:3,:,:]
        crop_mask_gt = (torch.sum(crop_imgs_colorcode_gt, dim=1, keepdim=True)>-0.9999*3).float()
        crop_mask_pred = (torch.sum(crop_img_colorcode_pred, dim=1, keepdim=True)>-0.95*3).float()
        crop_masks_contour = crop_masks_contour*crop_mask_gt
        if "symm" in colorcode_type:
            crop_symm_masks_pred = crop_img_colorcode_n_mask_pred[:,3:,:,:]
            crop_symm_masks_pred = torch.tanh(crop_symm_masks_pred)

        colorcode_diff_l1 = torch.abs(crop_img_colorcode_pred-crop_imgs_colorcode_gt)
        loss_cc1 = torch.sum(colorcode_diff_l1*crop_masks_contour*5+colorcode_diff_l1)/(3*batch_size*config["square_size"]**2)
        if "symm" in colorcode_type:
            # in mask loss + out mask loss
            loss_cc2_in = 3*torch.sum(crop_mask_gt)-torch.abs(torch.sum(crop_symm_masks_gt*crop_symm_masks_pred))
            loss_cc2_out = torch.sum(torch.abs(crop_symm_masks_gt-crop_symm_masks_pred)*(1-crop_mask_gt))
            loss_cc2 = (loss_cc2_in+loss_cc2_out)/(3*batch_size*config["square_size"]**2)
        else:
            loss_cc2 = torch.tensor([0], device=config["device"])
        loss_cc = loss_cc1+loss_cc2
        loss_perpixel = torch.sum(colorcode_diff_l1)/(3*batch_size*config["square_size"]**2)
        loss_perpixel_contour = torch.sum(colorcode_diff_l1*crop_masks_contour)/(torch.sum(crop_masks_contour)*3)

        sum_loss_cc1 += loss_cc1.item()*batch_size
        sum_loss_cc2 += loss_cc2.item()*batch_size
        sum_loss_perpixel += loss_perpixel.item()*batch_size
        sum_loss_perpixel_contour += loss_perpixel_contour.item()*batch_size

        loss_cc = loss_cc/config["accum_steps"]
        if mode == "train":
            loss_cc.backward()
            if (idx_batch+1) % config["accum_steps"] == 0 or (idx_batch+1) == len(dataloader):
                optimizer.step()
                

    avg_loss_cc1 = sum_loss_cc1/len(dataloader.dataset)
    avg_loss_cc2 = sum_loss_cc2/len(dataloader.dataset)
    avg_loss_cc = (sum_loss_cc1+sum_loss_cc2)/len(dataloader.dataset)
    avg_loss_perpixel = sum_loss_perpixel/len(dataloader.dataset)
    avg_loss_perpixel_contour = sum_loss_perpixel_contour/len(dataloader.dataset)

    # log message
    time_cost = time.time() - time_start
    message = "{} {} {} Epoch {:03d} time(s): {:.1f} loss_cc1: {:.6f}, loss_cc2: {:.6f}, loss_perpixel: {:.6f} loss_perpixel_contour: {:.6f}".format(
        idx_obj_str, mode, train_section, idx_epoch+1, time_cost, avg_loss_cc1, avg_loss_cc2, avg_loss_perpixel, avg_loss_perpixel_contour
    )
    logger.log(message)

    # save best model
    if mode == "valid":
        if avg_loss_perpixel_contour < logger.min_valid_loss:
            logger.min_valid_loss = avg_loss_perpixel_contour
            logger.best_model_epoch = idx_epoch+1
            torch.save(model_cc.state_dict(), os.path.join(config[weight_dir],idx_obj_str+"_best_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(config[weight_dir],idx_obj_str+"_best_optimizer.pth"))
            logger.log(idx_obj_str+" Best Model saved")

    return avg_loss_perpixel_contour

if __name__ == '__main__':
    # load configuration
    config_file = "config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    with open("config_obj.yaml", "r") as file:
        info_obj = yaml.safe_load(file)
    
    colorcode_type = "cc_aniso_n_symm"
    train_section = "colorcode"
    weight_dir = "weight_"+train_section+"_dir" 

    for idx_obj in [1,2,4,5,6,8,9,10,11,12,13,14,15]:
        idx_obj_str = str(idx_obj).zfill(6)

        # get the list of images
        with open(os.path.join("data list",idx_obj_str,"scene_train.txt"), 'r') as file:
            list_scene_train = [line.rstrip() for line in file.readlines()]
        with open(os.path.join("data list",idx_obj_str,"colorcode_train.txt"), 'r') as file:
            list_colorcode_train = [line.rstrip() for line in file.readlines()]
        with open(os.path.join("data list",idx_obj_str,"scene_valid.txt"), 'r') as file:
            list_scene_valid = [line.rstrip() for line in file.readlines()]
        with open(os.path.join("data list",idx_obj_str,"colorcode_valid.txt"), 'r') as file:
            list_colorcode_valid = [line.rstrip() for line in file.readlines()]

        # create dataloader
        mp.set_start_method('spawn', force=True)
        if "symm" in colorcode_type:
            load_symm_mask = True
        else:
            load_symm_mask = False
        dataloader_train = DataLoader(util.Dataset_colorcode(list_scene_train, list_colorcode_train, config, load_symm_mask=load_symm_mask), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=True)
        dataloader_valid = DataLoader(util.Dataset_colorcode(list_scene_valid, list_colorcode_valid, config, load_symm_mask=load_symm_mask), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=False)

        # create logger
        logger = util.Logger(config, idx_obj_str, weight_dir)

        # create sobel filter
        sobel = util.sobel_generator().to(config["device"])

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
        
        initialize_weights(model_seg)
        model_seg.load_state_dict(torch.load(os.path.join(config["weight_segmentation_dir"],idx_obj_str+"_best_model.pth")))
        model_seg.eval()
        initialize_weights(model_cc)
        # model_cc.load_state_dict(torch.load(os.path.join(config[weight_dir],idx_obj_str+"_best_model.pth")))

        # Optimizers
        optimizer = optim.Adam(model_cc.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-08)
        # optimizer.load_state_dict(torch.load(os.path.join(config[weight_dir],idx_obj_str+"_best_optimizer.pth")))

        alrs_scheduler = util.ALRS(optimizer, loss_ratio_threshold=0.02, decay_rate=0.98)
        model_noise_injector = util.Model_Noise_Injector(std_ratio=0.01, epoch_interval=10)

        # train and valid the network
        for idx_epoch in range(config["num_epochs"]):

            # train
            loss = run_epoch(model_seg, model_cc, "train", train_section, dataloader_train, optimizer, config, idx_epoch, logger)
            alrs_scheduler.step(loss)

            # valid
            if (idx_epoch+1) % 3 == 0:
                loss = run_epoch(model_seg, model_cc, "valid", train_section, dataloader_valid, optimizer, config, idx_epoch, logger)
                model_noise_injector.step(idx_epoch,model_cc,loss)


