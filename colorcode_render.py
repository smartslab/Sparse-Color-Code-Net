#%%
import time
import os
import shutil
import json
import sys
import yaml
import neural_renderer as nr
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pickle


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

def compute_ADD(pts3d, diameter, R_pred, t_pred, R_gt,t_gt):
    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.detach().cpu().numpy()
    if pts3d.shape[0] == 1:
        pts3d = pts3d.squeeze()

    R_pred = R_pred.reshape(3,3)
    t_pred = t_pred.reshape(3,1)
    R_gt = R_gt.reshape(3,3)
    t_gt = t_gt.reshape(3,1)
    pts_xformed_gt = np.matmul(R_gt.reshape(3,3),pts3d.transpose())+t_gt  
    pts_xformed_pred = np.matmul(R_pred,pts3d.transpose())+t_pred
    pts_xformed_diff = pts_xformed_gt-pts_xformed_pred
    distance = np.linalg.norm(pts_xformed_diff, axis=0)
    mean_distance = np.mean(distance)            
    ADD = mean_distance/diameter
    
    return ADD

def PnP(img, mask, vertices, idx_obj_str, info_obj, cam_K, colorcode_type, bbox=None):

    pix_pos = np.where(mask>0.1)
    pix_rgb = img[pix_pos[0],pix_pos[1]].astype(float)/255

    # apply for rotation correction for symmetric objects
    if "symm" in colorcode_type and len(info_obj[idx_obj_str]["R_correct"])!=0:
        rot_x, rot_y, rot_z = info_obj[idx_obj_str]["R_correct"]
        R_correct = xyz_euler_to_rotation_matrix(rot_x, rot_y, rot_z, mode='deg')
        R_correct_inv = torch.tensor(R_correct, dtype=vertices.dtype, device=vertices.device).unsqueeze(0).transpose(1, 2)
        vertices = torch.bmm(vertices, R_correct_inv)

    x_min = torch.min(vertices[0,:,0]).item()
    y_min = torch.min(vertices[0,:,1]).item()
    z_min = torch.min(vertices[0,:,2]).item()
    x_max = torch.max(vertices[0,:,0]).item()
    y_max = torch.max(vertices[0,:,1]).item()
    z_max = torch.max(vertices[0,:,2]).item()
    r_max = max([(x_max-x_min),(y_max-y_min),(z_max-z_min)])

    pix_2D = np.vstack((pix_pos[1],pix_pos[0])).T
    pix_2D = pix_2D.astype("double")
    if bbox is not None:
        size = img.shape[0] # consider the square shape
        cmin, rmin, cmax, rmax = bbox
        crop_height = rmax-rmin+1
        crop_width = cmax-cmin+1
        scale_factor = size/max(crop_height, crop_width)
        pix_2D = (pix_2D-size/2)/scale_factor
        pix_2D[:,0] = pix_2D[:,0]+(cmin+cmax)/2
        pix_2D[:,1] = pix_2D[:,1]+(rmin+rmax)/2

    #tbc
    if colorcode_type == "cc_aniso" or colorcode_type == "cc_aniso_n_symm":
        pix_3D = np.vstack((pix_rgb[:,0]*(x_max-x_min)+x_min,pix_rgb[:,1]*(y_max-y_min)+y_min,pix_rgb[:,2]*(z_max-z_min)+z_min)).T
    elif colorcode_type == "cc_origin":
        pix_3D = np.vstack((pix_rgb[:,0]*r_max+x_min,pix_rgb[:,1]*r_max+y_min,pix_rgb[:,2]*r_max+z_min)).T

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

def verify_pose(img,R_gt,t_gt):
    # verify the generated colorcode image
    img = img_cc_gt
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    mask = np.sum(img, axis=2)/255/3
    R_pred,t_pred = PnP(img,mask,vertices, idx_obj_str, info_obj, cam_K, colorcode_type)

def xyz_euler_to_rotation_matrix(x, y, z, mode='rad'):
    """
    Converts XYZ Euler angles to a rotation matrix.
    
    Parameters:
    x, y, z (float): Rotation angles around the x, y, and z axes.
    mode (str): The mode of angle measurement ('rad' for radians, 'deg' for degrees).

    Returns:
    numpy.ndarray: The corresponding rotation matrix.
    """
    if mode == 'deg':
        # Convert degrees to radians
        x, y, z = np.radians([x, y, z])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    # The rotation matrix is Rz * Ry * Rx
    return Rz @ Ry @ Rx

    
if __name__ == "__main__":

    with open("config_obj.yaml", "r") as file:
        info_obj = yaml.safe_load(file)

    img_height=480
    img_width=640
    list_colorcode_type = ["cc_origin","cc_aniso","cc_aniso_n_symm","symm_mask"]

    render_content = "pvnet"
    models_folder = "/content/drive/MyDrive/data/LINEMOD/models/"
    src_dir = "/content/renders"
    tgt_dir = "/content/LINEMOD_renders/base"
    # for LINEMOD
    # src_dir = "/content/LINEMOD/base"
    # tgt_dir = src_dir 

    if render_content!="pvnet": 

        for colorcode_type in list_colorcode_type:
            if colorcode_type == "symm_mask":
                renderer = nr.Renderer(img_height, img_width, camera_mode='projection',background_color=0.5)
            else:
                renderer = nr.Renderer(img_height, img_width, camera_mode='projection')

            for idx_obj_str in info_obj:
                print("working on object", idx_obj_str, colorcode_type)
                models_path = models_folder+"obj_"+idx_obj_str+".obj"
                vertices, faces, textures = obj_loader(idx_obj_str, info_obj, models_path, colorcode_type)
            
                with open(os.path.join(src_dir, idx_obj_str, "scene_gt.json")) as f:
                    scene_gt_data = json.load(f)
                with open(os.path.join(src_dir, idx_obj_str, "scene_camera.json")) as f:
                    scene_camera_data = json.load(f)

                # remove the old generated colorcode data
                if os.path.exists(os.path.join(tgt_dir, idx_obj_str, colorcode_type)):
                    shutil.rmtree(os.path.join(tgt_dir, idx_obj_str, colorcode_type))
                os.makedirs(os.path.join(tgt_dir, idx_obj_str, colorcode_type))

                for id_frame in scene_camera_data:
                    img_cc_path = os.path.join(tgt_dir, idx_obj_str, colorcode_type, id_frame.zfill(6)+".png")
                    list_obj_pose = scene_gt_data[id_frame]

                    for i in range(len(list_obj_pose)):
                        cam_K = np.array(scene_camera_data[id_frame]['cam_K']).reshape((3,3)).astype(np.float32)
                        R_gt = np.array(list_obj_pose[i]["cam_R_m2c"]).reshape((3,3)).astype(np.float32)
                        t_gt = np.array(list_obj_pose[i]["cam_t_m2c"]).reshape((3,1)).astype(np.float32)

                        if colorcode_type == "symm_mask":
                            textures1, textures2 = textures
                            f_rend_gt1, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures1, cam_K, R_gt, t_gt, renderer)
                            f_rend_gt2, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures2, cam_K, R_gt, t_gt, renderer)
                            f_rend_gt = 2*f_rend_gt1-f_rend_gt2
                        else:
                            f_rend_gt, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures, cam_K, R_gt, t_gt, renderer)
                        img_cc_gt = f_rend_gt.detach().cpu().numpy()[0,:3,:,:].transpose((1, 2, 0)).squeeze()*255
                        cv2.imwrite(img_cc_path,cv2.cvtColor(img_cc_gt, cv2.COLOR_RGB2BGR))

    else:
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        for colorcode_type in list_colorcode_type:
            if colorcode_type == "symm_mask":
                renderer = nr.Renderer(img_height, img_width, camera_mode='projection',background_color=0.5)
            else:
                renderer = nr.Renderer(img_height, img_width, camera_mode='projection')

            for idx_obj_str in info_obj:
                print("working on object", idx_obj_str, colorcode_type)
                models_path = models_folder+"obj_"+idx_obj_str+".obj"
                vertices, faces, textures = obj_loader(idx_obj_str, info_obj, models_path, colorcode_type)

                obj_name = info_obj[idx_obj_str]["name"]
                obj_src_dir = os.path.join(src_dir, obj_name)
                obj_tgt_dir = os.path.join(tgt_dir, idx_obj_str)

                if not os.path.exists(obj_tgt_dir):
                    os.makedirs(obj_tgt_dir)
                if not os.path.exists(os.path.join(obj_tgt_dir,"rgb")):
                    os.makedirs(os.path.join(obj_tgt_dir,"rgb"))
                if os.path.exists(os.path.join(obj_tgt_dir, colorcode_type)):
                    shutil.rmtree(os.path.join(obj_tgt_dir, colorcode_type))
                os.makedirs(os.path.join(obj_tgt_dir, colorcode_type))

                # count the number of files in obj_src_dir (png pkl jpg)
                num_all_files = len([f for f in os.listdir(obj_src_dir)if os.path.isfile(os.path.join(obj_src_dir, f))])
                num_files = int(num_all_files/3)
                if num_all_files%3!=0:
                    print("wrong number of files for object", obj_name)
                    sys.exit()

                # start processing each file
                scene_gt = {}
                scene_camera = {}
                for i in range(int(num_files)):

                    img_src_path = os.path.join(obj_src_dir, str(i).zfill(6)+".jpg")
                    img_tgt_path = os.path.join(obj_tgt_dir, str(i).zfill(6)+".jpg")
                    img_scene_src_path = os.path.join(obj_src_dir, str(i)+".jpg")
                    img_scene_tgt_path = os.path.join(obj_tgt_dir, "rgb", str(i).zfill(6)+".jpg")
                    if not os.path.exists(img_scene_tgt_path):
                        shutil.copy(img_scene_src_path, img_scene_tgt_path)

                    info_scene = pickle.load(open(os.path.join(obj_src_dir, str(i)+"_RT.pkl"), "rb"))
                    cam_K = info_scene["K"].astype(np.float32)
                    RT = info_scene["RT"]
                    R_gt = RT[:,0:3].astype(np.float32)
                    t_gt = RT[:,3:4].astype(np.float32)*1000

                    img_cc_tgt_path = os.path.join(obj_tgt_dir, colorcode_type, str(i).zfill(6)+".png")
                    if colorcode_type == "symm_mask":
                        textures1, textures2 = textures
                        f_rend_gt1, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures1, cam_K, R_gt, t_gt, renderer)
                        f_rend_gt2, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures2, cam_K, R_gt, t_gt, renderer)
                        f_rend_gt = 2*f_rend_gt1-f_rend_gt2
                    else:
                        f_rend_gt, face_index_map, depth_map = render_colorcode(img_height,img_width, vertices, faces, textures, cam_K, R_gt, t_gt, renderer)
                    img_cc_gt = f_rend_gt.detach().cpu().numpy()[0,:3,:,:].transpose((1, 2, 0)).squeeze()*255
                    cv2.imwrite(img_cc_tgt_path,cv2.cvtColor(img_cc_gt, cv2.COLOR_RGB2BGR))       
                    _ = 0

