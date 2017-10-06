import random
import numpy.linalg as la
import numpy as np
from matplotlib import pyplot as plt
#from ransac import *
 
def v_angles(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def run_ransac(data_raw, estimate, is_inlier, sample_size, goal_inliers, max_iterations,reference_vector, stop_at_goal=True, random_seed=None):
    
    """
    Demostrates projection of the velodyne points into the image plane
    Parameters
    ----------
    data_raw = data_set_velo
    estimate = regression function to find plane or line of best fit
    is_inlier = tolerance or how much deviance from regression line before considered inlier or outlier.
    base_dir  : Absolute path to sequence base directory (ends with _sync)
    calib_dir : Absolute path to directory that contains calibration files
    Returns
    -------
    """
    
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    data = data_raw[:,0:3]
    for i in range(max_iterations):
        
        # 1. Get samples of data
        s = data[np.random.randint(data.shape[0], size=sample_size), :]
        
        # 2. create regression line from samples
        m = estimate(s)
        ic = 0
        
        # 3. now iterate through all the data and see if close to regression line
        for j in range(len(data)):
            
            # to simplify can we check whether it is more than a certain z axis height
            #if abs(np.dot([0,0,1],data[j])) ==0 :
            
                if is_inlier(m, data[j]):
                    ic += 1
                    data_raw[j][3] = 1

                else:
                    data_raw[j][3] = 5
                    
            #else:
                #print(data[j])

        print ('estimate:', m)
        print ('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                a,b,c,d = m
                if v_angles(reference_vector,[a,b,c]) < 0.174533:
                    print ("angle %",v_angles([0,0,1],[a,b,c]))
                    break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic,data_raw


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    # performs single value decomposition
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold
    
def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[-20:80, -20:20]
    return xx, yy, (-d - a * xx - b * yy) / c
