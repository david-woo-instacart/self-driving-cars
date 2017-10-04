# Logic to convert velodyne coordinates to pixel coordinates

import numpy as np

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

l_and = lambda *x: np.logical_and.reduce(x)

def convert_velo_cord_to_img(data_set, calib_dir,num_points = 5, cam=2, frame=0,tracklet = False):
    """
    Demostrates projection of the velodyne points into the image plane
    Parameters
    ----------
    dataset = data_set_velo
    base_dir  : Absolute path to sequence base directory (ends with _sync)
    calib_dir : Absolute path to directory that contains calibration files
    Returns
    -------
    """
    calib = loadCalibrationCamToCam(calib_dir + 'calib_cam_to_cam.txt')
    Tr_velo_to_cam = loadCalibrationRigid(calib_dir + 'calib_velo_to_cam.txt')

    # Method
    # velodyne to camera coordiates through rigid body transformation (Tr)
    # camera coordinates to image coordinates through projection (P_rect)
    # image coordinates to pixel coordinates through rectification (R_rect)  
    R_cam_to_rect = np.eye(4, dtype=float)
    R_cam_to_rect[0: 3, 0: 3] = calib['R_rect_00']
    P_velo_to_img = np.dot(np.dot(calib['P_rect_0' + str(cam)], R_cam_to_rect), Tr_velo_to_cam)

    if tracklet: 
        velo_data = data_set
        velo = velo_data
    else:
        velo_data = data_set[frame]
        velo = velo_data[0:velo_data.shape[0]:num_points]
    
    #img_h, img_w, img_ch = dataset_rgb[frame].right.shape
    img_h, img_w, img_ch = 400,1500,3
    
    img_plane_depth = 5
    x_dir_pts = velo[:, 0]
    filtered_x_dir_indices = l_and((x_dir_pts > img_plane_depth))
#     .flatten to remove extra dimension
    indices = np.argwhere(filtered_x_dir_indices).flatten()
#     Depth (x) limited velodyne points
    velo = velo[indices, :]
    # Finally, Apply 2d projection function to velodyne points
    velo_img = project(velo[:, 0:3], P_velo_to_img)
    
    if tracklet:
        return velo_img
    else:
        return velo_img,velo
    

#main 3d points to camera projection function

def crop_velo_to_img_size(img_shape, calib_velo_data,velo_data_raw,include_z = False):
    """
    Parameters:
    ----------
    img_size: camera image size
    velo_data :calibrated and project transformed lidar to camera data
    """
    img_h = img_shape[0]
    img_w = img_shape[1]
    print("crop_velo velo_data %",calib_velo_data[:,0])
    img_dim_x_pts = calib_velo_data[:, 0]
    img_dim_y_pts = calib_velo_data[:, 1]
        
    x_filt = l_and((img_dim_x_pts < img_w), (img_dim_x_pts >= 0))
    y_filt = l_and((img_dim_y_pts < img_h), (img_dim_y_pts >= 0))
    filtered = l_and(x_filt, y_filt)
    indices = np.argwhere(filtered).flatten()
    
    img_dim_x_pts = img_dim_x_pts[indices]
    img_dim_y_pts = img_dim_y_pts[indices]
    
    if include_z:
        img_dim_z_pts = velo_data_raw
        
        print("indices %",indices.shape)
        
        img_dim_z_pts = img_dim_z_pts[indices]
        
        distance = img_dim_z_pts[:,0]
        
        print("using xaxis")xw
        return (img_dim_x_pts, img_dim_y_pts,distance)
    
    return (img_dim_x_pts, img_dim_y_pts)