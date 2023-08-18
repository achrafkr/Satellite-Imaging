# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 2023

@author: Achrafkr
"""

# Imports

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
import cv2

from image_tools import clustering_km, cv2_clustering_km, draw_contour, polygon_from_contour, compute_area

def segment_image(image_object, n_clusters, method = 'opencv', eps = 0.5, thresh = 0.75, t_method = 'binary'):
    
    if type(image_object) == str:
        img = plt.imread(image_object)
    
    elif type(image_object) == np.ndarray:
        img = image_object
        
    else:
        raise ValueError("Unknown object: use string or numpy.ndarray object instead")
        
    image_size = img.shape
        
    if method == 'sklearn':
        centroids, labels = clustering_km(img, n_clusters)
        segmented_img = np.choose(labels, centroids).astype(np.uint8)
        segmented_img.shape = image_size
        
    elif method == 'opencv':
        segmented_img = cv2_clustering_km(img, n_clusters, eps)
        
    grey_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    thresh = np.round(np.quantile(grey_img, thresh))
    
    contour_img, contours = draw_contour(segmented_img, thresh, t_method)
    
    return segmented_img, contour_img, contours



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    path_test = "../Data/EuroSat/2750/Test/"
    n_clusters = 3
    t_method = 'binary_inv' # threshold method
    image_filename = "AnnualCrop/AnnualCrop_1275.jpg"
    
    s_img, c_img, ctrs = segment_image(path_test + image_filename, n_clusters, t_method = t_method)
    
    
    fig, ax = plt.subplots(1, 3, figsize = (16, 4))

    ax[0].imshow(plt.imread(path_test + image_filename))
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    
    ax[1].imshow(s_img)
    ax[1].set_title('Segmented image')
    ax[1].axis('off')
    
    
    ax[2].imshow(c_img.astype(np.uint8))
    ax[2].set_title('Segmented image contour')
    ax[2].axis('off')
    
    plt.show()
    
    print("Sown cropped Area: {} ({:.2f} % of total land)".format(compute_area(ctrs), compute_area(ctrs, True, s_img.shape)))
    

    
    
    
    
    
    
    
    
    
    
    
    