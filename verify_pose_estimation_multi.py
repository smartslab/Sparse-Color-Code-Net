import os
import cv2
import shutil
import yaml
import torch
import torch.nn as nn
import colorcode_util as util
import matplotlib.pyplot as plt
from colorcode_model import UNet, initialize_weights, FocalLoss, TverskyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import gc
import numpy as np

# todo (for output, 0 is grey, -1 is black)
def run_epoch(model_seg, model_cc, mode, train_section, dataloader, optimizer, config, idx_epoch, logger):
    
    model_seg.eval()
    model_cc.eval()
    torch.set_grad_enabled(False)

    counter = 0
    sum_loss_seg = 0
    sum_loss_tversky = 0
    sum_loss_focal = 0


    for idx_batch, data in enumerate(dataloader):
        if "symm" in colorcode_type:
            imgs_scene, imgs_colorcode_gt, masks_gt, symm_masks = data
        else:
            imgs_scene, imgs_colorcode_gt, masks_gt = data

        batch_size = imgs_scene.shape[0]
        imgs_scene = imgs_scene.to(config["device"])
        imgs_colorcode_gt = imgs_colorcode_gt.to(config["device"])
        masks_gt = masks_gt.to(config["device"])
        if "symm" in colorcode_type:
            symm_masks = symm_masks.to(config["device"])

        # get the sparse contour of the image todo
        masks_contour = util.generate_mask_contour(sobel,imgs_scene)
        masks_gt_obj = torch.sum(masks_gt[:,:-1,:,:], dim=1, keepdim=True)
        
        # Forward pass through the main network
        imgs_scene_n_contour = torch.cat((imgs_scene, masks_contour), dim=1)
        masks_raw, fullsize_feature = model_seg(imgs_scene_n_contour)
        masks_channel = (masks_gt.sum(dim=(0, 2, 3), keepdim=True)>0).float() 
        masks_prob = F.softmax(masks_raw, dim=1)
        masks_pred = (masks_prob>0.5).float()

        imgs_scene_n_contour_n_mask = torch.cat((imgs_scene, masks_contour, masks_gt[:,:-1,:,:]), dim=1)

        crop_input = torch.zeros((batch_size, imgs_scene_n_contour_n_mask.shape[1], config["square_size"], config["square_size"]), device=config["device"])
        crop_imgs_colorcode_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])
        crop_masks_contour = torch.zeros((batch_size, 1, config["square_size"], config["square_size"]), device=config["device"])
        if "symm" in colorcode_type:
            crop_symm_masks_gt = torch.zeros((batch_size, 3, config["square_size"], config["square_size"]), device=config["device"])

        ######################################################
        # plot these masks as 3x3 figure using plt
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        for i in range(3):
            for j in range(3):
                mask = np.zeros((masks_prob.shape[2], masks_prob.shape[3], 3))
                mask[:,:,0] = masks_gt[0,i*3+j,:,:].cpu().numpy()
                mask[:,:,1] = masks_pred[0,i*3+j,:,:].cpu().numpy()
                ax[i, j].imshow(mask)
        # save the figure
        plt.savefig('masks_compare.png')
        # save the imgs_scene as well
        # start a new figure
        # plt.figure()
        # img_scene = (imgs_scene[0,:,:,:].permute(1, 2, 0).cpu().numpy()+1)/2
        # print("img_scene min max", img_scene.min(), img_scene.max())
        # plt.imshow(img_scene)
        # plt.savefig('imgs_scene.png')
        ######################################################

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

        crop_mask_gt = crop_input[:,1-config["num_classes"]:,:,:]
        crop_img_colorcode_n_mask_pred, _ = model_cc(crop_input)
        crop_img_colorcode_pred = crop_img_colorcode_n_mask_pred[:,:3,:,:]
        crop_masks_contour = crop_masks_contour*crop_mask_gt
        if "symm" in colorcode_type:
            crop_symm_masks_pred = crop_img_colorcode_n_mask_pred[:,3:,:,:]
            crop_symm_masks_pred = torch.tanh(crop_symm_masks_pred)

    return 0

# load configuration
if __name__ == '__main__':
    config_file = "config_multi.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    train_section = "segmentation"
    weight_dir = "weight_"+train_section+"_dir"

    # create sobel filter
    sobel = util.sobel_generator().to(config["device"])

    with open(os.path.join("data list","scene_valid_multi.txt"), 'r') as file:
        list_scene_valid = [line.rstrip() for line in file.readlines()]
    with open(os.path.join("data list","mask_valid_multi.txt"), 'r') as file:
        list_colorcode_valid = [line.rstrip() for line in file.readlines()]

    # create dataloader
    mp.set_start_method('spawn', force=True)
    dataloader_valid = DataLoader(util.Dataset_colorcode_with_mask(list_scene_valid, list_colorcode_valid, config, load_symm_mask=load_symm_mask), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=False)

    # create logger
    idx_obj_str = "multi"
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
    
    initialize_weights(model_seg)
    initialize_weights(model_cc)
    model_seg.load_state_dict(torch.load(os.path.join(config[weight_dir],"multi_seg_best_model.pth")))
    model_cc.load_state_dict(torch.load(os.path.join(config[weight_dir],"multi_cc_best_model.pth")))
    optimizer = torch.optim.Adam(model_seg.parameters(), lr=0.1)

    loss = run_epoch(model_seg, model_cc, "valid", train_section, dataloader_valid, optimizer, config, 0, logger)





