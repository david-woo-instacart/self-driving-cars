# Logic to convert velodyne coordinates to pixel coordinates

import numpy as np
import cv2 as cv2
import statistics
import sys
import numpy as np
import pykitti
import matplotlib.pyplot as plt

from source import parseTrackletXML as xmlParser
from source import dataset_utility as du

#### Helper function ####

def parse_string_variable(str):
    var_name = str.split(':')[0]
    after_colon_index = len(var_name) + 1
    value = str[after_colon_index:]
    return (var_name, value)

def read_lines_to_dict(raw_text):
    var_list = []
    for i, line in enumerate(raw_text):
        var_list.append(line.replace('\n', ''))
    for i, line in enumerate(raw_text):
        var_list[i] = parse_string_variable(line)
    return dict(var_list)

def read_files_by_lines(filename):
    assert type(filename) is str
    with open(filename, 'r') as cam_to_cam:
        data = cam_to_cam.readlines()
    return read_lines_to_dict(data)

def replace_var_from_dict_with_shape(var_dict, key, shape):
    return np.array(var_dict[key]).reshape(shape)

#### Load calibration files ####

def loadCalibrationCamToCam(filename, verbose=False):
    assert type(filename) is str
    cam_dict = read_files_by_lines(filename)

    for key, value in cam_dict.items():
        if key == 'calib_time':
            cam_dict[key] = value
        else:
            array = []
            for i, string in enumerate(value.split(' ')[1:]):
                array.append(float(string))
            cam_dict[key] = array

    for i in range(0, 4):
        S_rect_0i = 'S_rect_0' + str(i)
        R_rect_0i = 'R_rect_0' + str(i)
        P_rect_0i = 'P_rect_0' + str(i)
        S_0i = 'S_0' + str(i)
        K_0i = 'K_0' + str(i)
        D_0i = 'D_0' + str(i)
        R_0i = 'R_0' + str(i)
        T_0i = 'T_0' + str(i)

        cam_dict[S_rect_0i] = replace_var_from_dict_with_shape(cam_dict, S_rect_0i, (1, 2))
        cam_dict[R_rect_0i] = replace_var_from_dict_with_shape(cam_dict, R_rect_0i, (3, 3))
        cam_dict[P_rect_0i] = replace_var_from_dict_with_shape(cam_dict, P_rect_0i, (3, 4))
        cam_dict[S_0i] = replace_var_from_dict_with_shape(cam_dict, S_0i, (1, 2))
        cam_dict[K_0i] = replace_var_from_dict_with_shape(cam_dict, K_0i, (3, 3))
        cam_dict[D_0i] = replace_var_from_dict_with_shape(cam_dict, D_0i, (1, 5))
        cam_dict[R_0i] = replace_var_from_dict_with_shape(cam_dict, R_0i, (3, 3))
        cam_dict[T_0i] = replace_var_from_dict_with_shape(cam_dict, T_0i, (3, 1))

    if verbose:
          print(S_rect_0i, cam_dict[S_rect_0i])
          print(R_rect_0i, cam_dict[R_rect_0i])
          print(P_rect_0i, cam_dict[P_rect_0i])
          print(S_0i, cam_dict[S_0i])
          print(K_0i, cam_dict[K_0i])
          print(D_0i, cam_dict[D_0i])
          print(R_0i, cam_dict[R_0i])
          print(T_0i, cam_dict[T_0i])
    return cam_dict

def loadCalibrationRigid(filename, verbose=False):
    assert type(filename) is str
    velo_dict = read_files_by_lines(filename)

    for key, value in velo_dict.items():
        if key == 'calib_time':
            velo_dict[key] = value
        else:
            array = []
            for i, string in enumerate(value.split(' ')[1:]):
                array.append(float(string))
            velo_dict[key] = array

    R = 'R'
    T = 'T'
    velo_dict[R] = replace_var_from_dict_with_shape(velo_dict, R, (3, 3))
    velo_dict[T] = replace_var_from_dict_with_shape(velo_dict, T, (3, 1))

    Tr = np.vstack((np.hstack((velo_dict[R], velo_dict[T])), [0, 0, 0, 1]))
    velo_dict['Tr'] = Tr

    if verbose:
      print(R, velo_dict[R])
      print(T, velo_dict[T])
      print('Tr', velo_dict['Tr'])
    return velo_dict['Tr']

def get_rigid_body_transformation(calib_file):
    
    #step 1: Getting rigid body transformation seems easy
    T_cam_velo = loadCalibrationRigid(calib_file)
    
    return T_cam_velo

def convert_to_rgb(minval, maxval, val, colors):
    
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i

    (r1, g1, b1), (r2, g2, b2) = colors[1], colors[2]
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

velodyne_max_x=100  # this scales the x-axis values. maybe replace with range

# Overlay velodyne points over image
def overlay_velo_img(img, velo_data,radius = 2):
    (x, y,z) = velo_data
    im = np.zeros(img.shape, dtype=np.float32)
    x_axis = np.floor(x).astype(np.int32)
    y_axis = np.floor(y).astype(np.int32)
    z_axis = np.floor(z).astype(np.int32)

    # below draws circles on the image
    for i in range(0, len(x)):

        if z_axis[i] <= 0:
            color = int(1/velodyne_max_x * 256)
            value = 0
        else:
            color = int((z_axis[i])/velodyne_max_x * 256)
            value = z_axis[i] * 4
        colors_range = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
        r, g, b  = convert_to_rgb(0,256,value,colors_range) 
        
        cv2.circle(img, (x_axis[i], y_axis[i]), radius, [r, g, b],-1)

    fig1 = plt.figure(figsize=(20, 20))
    
    return img

#### Main functions to perform the projection ####

def project(p_in, T):
    """
    Parameters
    ----------
    p_in  = velodyne points
    T = velodyne to pixel coordinates transformation
    Returns
    -------
    """
    
    dim_norm, dim_proj = T.shape

    p_in_row_count = p_in.shape[0]
    
#   Do transformation in homogenouous coordinates
    
    p2_in = p_in
    if p2_in.shape[1] < dim_proj:
        col_ones = np.ones(p_in_row_count)
        col_ones.shape = (p_in_row_count, 1)

        p2_in = np.hstack((p2_in, col_ones))
#   (T*p2_in')'
    p2_out = np.transpose(np.dot(T, np.transpose(p2_in)))
#   Normalize homogeneous coordinates
    denominator = np.outer(p2_out[:, dim_norm - 1], np.ones(dim_norm - 1))
#   Element wise division
    p_out = p2_out[:, 0: dim_norm-1]/denominator
    return p_out

#main 3d points to camera projection function
def convert_velo_cord_to_img_cord_test(velo_data,calib_dir,cam = 2,tracklet = False):
    
    #step 1: velo to camera coordinates Getting rigid body transformation seems easy
    T_cam_velo = get_rigid_body_transformation(calib_dir + 'calib_velo_to_cam.txt' )
    
    #step 2: camera coordinate to image coordinate. projection
    calib = loadCalibrationCamToCam(calib_dir + 'calib_cam_to_cam.txt')

    R_cam_to_rect = np.eye(4, dtype=float)
    R_cam_to_rect[0: 3, 0: 3] = calib['R_rect_00']
    
    #step 3: Create matrix to do RBT, projection, rectification
    transform_matrix = np.dot(np.dot(calib['P_rect_0' + str(cam)],R_cam_to_rect),T_cam_velo)
    
    if tracklet:
        return project(velo_data,transform_matrix)
    else:
        return project(velo_data,transform_matrix),velo_data
    

#main 3d points to camera projection function

intersect = lambda *x: np.logical_and.reduce(x)
def crop_to_img_size(img_size,projected_data,original_data):

    """
    Parameters:
    ----------
    img_size: camera image size
    projected_data : transformed lidar to camera data
    original_data : raw velodyne data
    original_data : include depth metrics in lidar ( x-coordinates in velodyne)
    
    Output:
    cropped_x_points
    cropped_y_points
    depth_points
    """
    
    img_h = img_size[0]
    img_w = img_size[1]
    
    # step 1: get indexes coordinates are outside of camera image size. i.e more than the length of the camera image
    filter_x = intersect((projected_data[:,0]<img_w),(projected_data[:,0]>=0))
    filter_y = intersect((projected_data[:,1]<img_h),(projected_data[:,1]>=0))
    comb_filter = intersect(filter_x,filter_y)
    
    #find indices where both x and y are within image size
    indices = np.argwhere(comb_filter).flatten()
    
    img_dim_x_pts = projected_data[:,0][indices]
    img_dim_y_pts = projected_data[:,1][indices]
    
    img_dim_dept_pts = original_data[:,0][indices]
    
    return (img_dim_x_pts, img_dim_y_pts,img_dim_dept_pts)