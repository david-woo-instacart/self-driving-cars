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

def estimate(xyzs):
    # performs single value decomposition
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def ransac(velo_data, inlier_threshold, sample_size, goal_inliers, max_iterations,reference_vector, stop_at_goal=True, random_seed=None):
    
    """
    Demostrates projection of the velodyne points into the image plane
    Parameters
    ----------
    velo_data = velo data points
    inlier_threshold = tolerance or how much deviance from regression line before considered inlier or outlier.
    
    Returns
    -------
    best_model = coefficients of best model
    best_ic = best number of inliers
    velo_data = velo data with points marked as inliers vs outliers
    """
    
    #Steps
    
    #1. Interate through max iterations
    
    data = velo_data[:,0:3]
    best_inliers_ct = 0
    
    for i in range(0,max_iterations):
        
        #2. Get sample of velo point
        s = data[np.random.randint(data.shape[0],size = sample_size),:]
        
        #3. Get line or plane of best fit. return coefficients of line of best fit        
        coeffs = estimate(s)
        
        inliers_ct = 0
        
        #4. Add the rest of the data points
        for j in range(len(data)):
            
            #5. check if data points are close to plane
            if is_inlier(coeffs, data[j], inlier_threshold):
                inliers_ct += 1
                velo_data[j][3] = 1
            else:
                velo_data[j][3] = 5
                
                
        if inliers_ct > best_inliers_ct:
            best_ic = inliers_ct
            best_model = coeffs
            if inliers_ct > goal_inliers and stop_at_goal:
                a,b,c,d = coeffs
                # this is approximately 10 degrees
                if v_angles(reference_vector,[a,b,c]) < 0.174533:
                    print ("angle %",v_angles([0,0,1],[a,b,c]))
                    break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic,velo_data

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold
    
def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[-20:80, -20:20]
    return xx, yy, (-d - a * xx - b * yy) / c
