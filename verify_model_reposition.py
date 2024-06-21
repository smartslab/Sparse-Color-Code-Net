#%%
import time
import os
import shutil
import json
import sys
import trimesh
import neural_renderer as nr
import pandas as pd
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio
import scipy.ndimage
import random
import pickle
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift
from colorcode_render import xyz_euler_to_rotation_matrix

def obj_loader(obj_name,render_colorcode,R_correct=None, T_correct=None, symm=False):
    vertices, faces, textures  = nr.load_obj(obj_name)
    vertices = torch.reshape(vertices, (1,vertices.shape[0],3)).cuda()

    vertices_default = vertices.clone()
    # move the vertices back to default pos corresponds to ground truth extrinsics
    if R_correct is not None and T_correct is not None:
        # vertices = vertices[0,:,:].detach().cpu().numpy()
        # vertices = vertices + np.dot(T_correct,R_correct)
        # vertices = np.dot(vertices,R_correct.T)
        # vertices = torch.from_numpy(vertices).cuda()
        # vertices = torch.reshape(vertices, (1,vertices.shape[0],3))
        # vertices = vertices.float()

        R_correct_tensor = torch.tensor(R_correct, dtype=vertices.dtype, device=vertices.device).unsqueeze(0)
        vertices = torch.bmm(vertices, R_correct_tensor.transpose(1, 2))

    faces = torch.reshape(faces, (1,faces.shape[0],3)).cuda()
    textures = torch.zeros(vertices.shape).cuda()

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

    if render_colorcode == "colorcode anisotropic":
      for idx_f in range(textures.shape[1]):
          textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/(x_max-x_min)
          if symm:
              textures[0,idx_f,1] = torch.abs((vertices[0,idx_f,1]-y_mid)/(y_max-y_min)*2)
          else:
              textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/(y_max-y_min)
          
          textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/(z_max-z_min)
    else:
        print("wrong style input")
        sys.exit()

    return vertices_default, faces, textures

def obj_loader_for_repos(obj_name,render_colorcode,R_correct=None, T_correct=None, symm=False):
    vertices, faces, textures  = nr.load_obj(obj_name)

    vertices = torch.reshape(vertices, (1,vertices.shape[0],3)).cuda()
    faces = torch.reshape(faces, (1,faces.shape[0],3)).cuda()
    textures = torch.zeros(vertices.shape).cuda()

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

    if render_colorcode == "colorcode anisotropic":
      for idx_f in range(textures.shape[1]):
          textures[0,idx_f,0] = (vertices[0,idx_f,0]-x_min)/(x_max-x_min)
          if symm:
              textures[0,idx_f,1] = torch.abs((vertices[0,idx_f,1]-y_mid)/(y_max-y_min)*2)
          else:
              textures[0,idx_f,1] = (vertices[0,idx_f,1]-y_min)/(y_max-y_min)
          
          textures[0,idx_f,2] = (vertices[0,idx_f,2]-z_min)/(z_max-z_min)
    else:
        print("wrong style input")
        sys.exit()

    # move the vertices back to default pos corresponds to ground truth extrinsics
    if R_correct is not None and T_correct is not None:
        vertices = vertices[0,:,:].detach().cpu().numpy()
        vertices = np.dot(vertices,R_correct)
        vertices = vertices - np.dot(T_correct,R_correct)
        vertices = torch.from_numpy(vertices).cuda()
        vertices = torch.reshape(vertices, (1,vertices.shape[0],3))
        vertices = vertices.float()

    return vertices, faces, textures

def render_colorcode(img_height,img_width, vertices, faces, textures, K, R, t, renderer):

    K = K.reshape((1,3,3)).astype(np.float32)
    K = torch.from_numpy(K).cuda()
    t = t.reshape((1,1,3)).astype(np.float32)
    t = torch.from_numpy(t).cuda()
    
    if torch.is_tensor(R):
        _ = 0
    else:
        R = R.reshape((1,3,3)).astype(np.float32)
        R = torch.from_numpy(R).cuda()

    f_rend, face_index_map, depth_map = renderer(vertices, faces, textures, K, R, t, 0, 0)
  

    return f_rend, face_index_map, depth_map

def PnP(img,vertices,cam_K,render_colorcode_mode,R_correct, T_correct, symm):
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)

    if R_correct is not None and T_correct is not None:
        # vertices = vertices[0,:,:].detach().cpu().numpy()
        # vertices = vertices + np.dot(T_correct,R_correct)
        # vertices = np.dot(vertices,R_correct.T)
        # vertices = torch.from_numpy(vertices).cuda()
        # vertices = torch.reshape(vertices, (1,vertices.shape[0],3))
        # vertices = vertices.float()

        R_correct_inv = torch.tensor(R_correct, dtype=vertices.dtype, device=vertices.device).unsqueeze(0).transpose(1, 2)
        vertices = torch.bmm(vertices, R_correct_inv)

    x_min = torch.min(vertices[0,:,0]).item()
    y_min = torch.min(vertices[0,:,1]).item()
    z_min = torch.min(vertices[0,:,2]).item()
    x_max = torch.max(vertices[0,:,0]).item()
    y_max = torch.max(vertices[0,:,1]).item()
    z_max = torch.max(vertices[0,:,2]).item()

    r_max = max([(x_max-x_min),(y_max-y_min),(z_max-z_min)])

    pix_pos = np.where(mask>0.9)
    pix_rgb = img[pix_pos[0],pix_pos[1]].astype(float)/255

    pix_2D = np.vstack((pix_pos[1],pix_pos[0])).T
    pix_2D = pix_2D.astype("double")

    if render_colorcode_mode == "colorcode anisotropic":
        pix_3D = np.vstack((pix_rgb[:,0]*(x_max-x_min)+x_min,pix_rgb[:,1]*(y_max-y_min)+y_min,pix_rgb[:,2]*(z_max-z_min)+z_min)).T
    elif render_colorcode_mode == "colorcode isotropic":
        pix_3D = np.vstack((pix_rgb[:,0]*r_max+x_min,pix_rgb[:,1]*r_max+y_min,pix_rgb[:,2]*r_max+z_min)).T

    if R_correct is not None and T_correct is not None:
        pix_3D = np.dot(pix_3D, R_correct)
        pix_3D = pix_3D - np.dot(T_correct,R_correct)

    # if num_sample!=0:
    #     idx_select = random.sample(range(len(pix_2D)), num_sample)
    #     pix_3D = pix_3D[idx_select,:]
    #     pix_2D = pix_2D[idx_select,:]

    ret, rvec, tvec,inliers = cv2.solvePnPRansac(pix_3D, pix_2D, cam_K,None,reprojectionError=5,iterationsCount=100)

    R_pred = np.eye(3)
    t_pred = tvec[:,0].reshape(3,1)
    cv2.Rodrigues(rvec, R_pred)  
    return R_pred,t_pred

if __name__ == "__main__":
    
    img_height = int(480)
    img_width = int(640)
    renderer = nr.Renderer(img_height, img_width, camera_mode='projection')

    cam_K = np.array([[700,0,320],[0,700,240],[0,0,1]]).astype(np.float64)
    R_gt = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float64)
    t_gt = np.array([0,0,800]).astype(np.float64)

    symm = False

    vertices, faces, textures = obj_loader("model_repos/origin.obj","colorcode anisotropic",R_correct=None, T_correct=None, symm=symm)
    f_rend_gt, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures, cam_K, R_gt, t_gt, renderer)
    img_cc = f_rend_gt.detach().cpu().numpy()[0,:3,:,:].transpose((1, 2, 0)).squeeze()*255
    cv2.imwrite("model_repos/origin.png",cv2.cvtColor(img_cc, cv2.COLOR_RGB2BGR))
    R_pred_origin,t_pred_origin = PnP(img_cc,vertices,cam_K,"colorcode anisotropic",R_correct=None, T_correct=None, symm=symm)

    R_correct = xyz_euler_to_rotation_matrix(-25.148, 5.7671, -21.648, mode='deg')
    # T_correct = np.array([-138.04,-65.632,0]).astype(np.float64)
    T_correct = np.array([0,0,0]).astype(np.float64)  # T can be ignored as it will be compensated by relative coordinate calculation

    vertices, faces, textures = obj_loader("model_repos/origin.obj","colorcode anisotropic",R_correct=R_correct,T_correct=T_correct,symm=symm)
    f_rend_gt, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures, cam_K, R_gt, t_gt, renderer)
    img_cc = f_rend_gt.detach().cpu().numpy()[0,:3,:,:].transpose((1, 2, 0)).squeeze()*255
    print("img_cc max min",np.max(img_cc),np.min(img_cc))
    cv2.imwrite("model_repos/origin2repos.png",cv2.cvtColor(img_cc, cv2.COLOR_RGB2BGR))
    R_pred_origin2repos,t_pred_origin2repos = PnP(img_cc,vertices,cam_K,"colorcode anisotropic",R_correct=R_correct, T_correct=T_correct, symm=symm) 

    vertices, faces, textures = obj_loader_for_repos("model_repos/repos.obj","colorcode anisotropic",R_correct=R_correct,T_correct=T_correct,symm=symm)
    f_rend_gt, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures, cam_K, R_gt, t_gt, renderer)
    img_cc = f_rend_gt.detach().cpu().numpy()[0,:3,:,:].transpose((1, 2, 0)).squeeze()*255
    cv2.imwrite("model_repos/repos.png",cv2.cvtColor(img_cc, cv2.COLOR_RGB2BGR))
    R_pred_repos,t_pred_repos = PnP(img_cc,vertices,cam_K,"colorcode anisotropic",R_correct=R_correct, T_correct=T_correct, symm=symm) 


    
    


            
                

