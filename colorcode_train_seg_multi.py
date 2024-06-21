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

# todo (for output, 0 is grey, -1 is black)
def run_epoch(model_seg, mode, train_section, dataloader, optimizer, config, idx_epoch, logger):
    
    if mode == "train":
        model_seg.train()
        torch.set_grad_enabled(True)
    elif mode == "valid" or mode == "record":
        model_seg.eval()
        torch.set_grad_enabled(False)

    counter = 0
    sum_loss_seg = 0
    sum_loss_tversky = 0
    sum_loss_focal = 0

    time_start = time.time()

    for idx_batch, (imgs_scene, masks_gt) in enumerate(dataloader):
        batch_size = imgs_scene.shape[0]
        imgs_scene = imgs_scene.to(config["device"])
        masks_gt = masks_gt.to(config["device"])

        # # ################################################
        # # plot these masks as 3x3 figure using plt
        # fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        # for i in range(3):
        #     for j in range(3):
        #         ax[i, j].imshow(masks_gt[0,i*3+j,:,:].cpu().numpy())
        # # save the figure
        # plt.savefig('masks_gt.png')
        # # save the imgs_scene as well
        # # start a new figure
        # plt.figure()
        # img_scene = (imgs_scene[0,:,:,:].permute(1, 2, 0).cpu().numpy()+1)/2
        # print("img_scene min max", img_scene.min(), img_scene.max())
        # plt.imshow(img_scene)
        # plt.savefig('imgs_scene.png')
        # # ################################################
        
        # get the sparse contour of the image todo
        masks_contour = util.generate_mask_contour(sobel,imgs_scene)
        masks_label = torch.argmax(masks_gt, dim=1, keepdim=False)
        
        # Forward pass through the main network
        imgs_scene_n_contour = torch.cat((imgs_scene, masks_contour), dim=1)
        masks_raw, fullsize_feature = model_seg(imgs_scene_n_contour)
        masks_channel = (masks_gt.sum(dim=(0, 2, 3), keepdim=True)>0).float() 
        masks_prob = F.softmax(masks_raw, dim=1)

        if torch.sum(masks_channel)==2: # single object data
            loss_focal = torch.tensor(0.0).to(config["device"])
            idx_channel = torch.nonzero(masks_channel[0,:,0,0])[0]
            loss_tversky = TverskyLoss(masks_raw[:,idx_channel,:,:], masks_gt[:,idx_channel,:,:], alpha=2, beta=0.5)
            _ = 0
        else:
            loss_focal = FocalLoss(masks_raw, masks_label, gamma=2)
            loss_tversky = TverskyLoss(masks_raw, masks_gt, alpha=2, beta=0.5)
        loss_seg = loss_focal + loss_tversky

        sum_loss_seg += loss_seg.item()*batch_size
        sum_loss_tversky += loss_tversky.item()*batch_size
        sum_loss_focal += loss_focal.item()*batch_size
        
        # Backward pass and optimization
        loss_seg = loss_seg/config["accum_steps"]
        if mode == "train":
            loss_seg.backward()
            if (idx_batch+1) % config["accum_steps"] == 0 or (idx_batch+1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

    avg_loss_seg = sum_loss_seg/len(dataloader.dataset)
    avg_loss_tversky = sum_loss_tversky/len(dataloader.dataset)
    avg_loss_focal = sum_loss_focal/len(dataloader.dataset)
    message = "{} {} {} Epoch {:03d} time(s): {:.1f} loss_tversky {:.9f} loss_focal {:.9f} loss_seg {:.9f}".format(
        idx_obj_str, mode, train_section, idx_epoch+1, time.time() - time_start, avg_loss_tversky,avg_loss_focal,avg_loss_seg
    )
    logger.log(message)

    # save best model
    if mode == "valid":
        if avg_loss_seg < logger.min_valid_loss:
            logger.min_valid_loss = avg_loss_seg
            logger.best_model_epoch = idx_epoch+1
            torch.save(model_seg.state_dict(), os.path.join(config[weight_dir],"multi_seg_best_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(config[weight_dir],"multi_seg_best_optimizer.pth"))
            logger.log("Best seg-multi Model saved")

    return avg_loss_seg

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
    dataloader_valid = DataLoader(util.Dataset_colorcode_multi(list_scene_valid, list_colorcode_valid, config, load_multiple=True), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=False)

    # create logger
    logger = util.Logger(config, "multi", weight_dir)

    # initialize the network and optimizer
    model_seg = UNet(num_depth=config["num_depth_scene"], 
                    num_basefilter=config["num_basefilter_scene"], 
                    input_channels=config["input_channels_scene"], 
                    output_channels=config["num_classes"],
                    kernel_size_down = config["kernel_size_scened_down"],
                    kernel_size_up = config["kernel_size_scened_up"],
                    ).to(config["device"])
    
    initialize_weights(model_seg)
    optimizer = torch.optim.Adam(model_seg.parameters(), lr=0.1)
    alrs_scheduler = util.ALRS(optimizer, loss_ratio_threshold=0.03, decay_rate=0.97)
    model_noise_injector = util.Model_Noise_Injector(std_ratio=0.01, epoch_interval=6)

    # train and valid the network
    for idx_epoch in range(config["num_epochs"]):

        # train

        # train multi
        if True: 
            idx_obj_str = "multi"
            # with open(os.path.join("data list","scene_train_multi_fuse.txt"), 'r') as file:
            #     list_scene_train = [line.rstrip() for line in file.readlines()]
            # with open(os.path.join("data list","mask_train_multi_fuse.txt"), 'r') as file:
            #     list_colorcode_train = [line.rstrip() for line in file.readlines()]
            with open(os.path.join("data list","scene_valid_multi_refine.txt"), 'r') as file:
                list_scene_train = [line.rstrip() for line in file.readlines()]
            with open(os.path.join("data list","mask_valid_multi_refine.txt"), 'r') as file:
                list_colorcode_train = [line.rstrip() for line in file.readlines()]
            dataloader_train = DataLoader(util.Dataset_colorcode_multi(list_scene_train, list_colorcode_train, config, load_multiple=True), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=True)
            loss = run_epoch(model_seg, "train", train_section, dataloader_train, optimizer, config, idx_epoch, logger)

        # # train single
        # if idx_epoch>10:
        #     for i in range(8): 
        #         list_obj =  [1,5,6,8,9,10,11,12]
        #         idx_obj_str = str(list_obj[i]).zfill(6)

        #         with open(os.path.join("data list",idx_obj_str,"scene_refine.txt"), 'r') as file:
        #             list_scene_train = [line.rstrip() for line in file.readlines()]
        #         with open(os.path.join("data list",idx_obj_str,"colorcode_refine.txt"), 'r') as file:
        #             list_colorcode_train = [line.rstrip() for line in file.readlines()]
        #         dataloader_train = DataLoader(util.Dataset_colorcode_multi(list_scene_train, list_colorcode_train, config, load_multiple=False, idx_obj=i), num_workers=config["num_workers"], batch_size=config["batch_size"], shuffle=True)       
        #         run_epoch(model_seg, "train", train_section, dataloader_train, optimizer, config, idx_epoch, logger)
        
        alrs_scheduler.step(loss)

        # valid
        if (idx_epoch+1) % 3 == 0:
            idx_obj_str = "multi"
            loss = run_epoch(model_seg, "valid", train_section, dataloader_valid, optimizer, config, idx_epoch, logger)
            model_noise_injector.step(idx_epoch,model_seg,loss)




