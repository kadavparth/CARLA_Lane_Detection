import numpy as np 
import cv2


def extract_features(image):

    """
    Extract ORB descriptors from an image 
    """

    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(image, None)

    return kp , des 


def visualize_features(image, kp):

    """
    Visualise extracted features in the image 

    image -- a grayscale image 
    kp -- list of the extracted keypoints
    
    Returns:
    """

    display = cv2.drawKeypoints(image, kp, None)
    
    return display


def extract_features_dataset(images, extract_features_function):

    """
    
    extract important features and return list of key points (kp) and a 
    list of descriptors (des_list) for each image 
    
    """
    
    kp_list = []
    des_list = []
    
    for image in images:
        kp , des = extract_features_function(image)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


def match_features(des1, des2):

    """
    Match features from two images

    des1 : list of keypoint descriptors in first image 
    des2 : list of keypoint descriptors in second image

    Returns : list of matched features from two images 
    """

    FlANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FlANN_INDEX_KDTREE, trees = 16)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match = flann.knnMatch(np.float32(des1), np.float32(des2), k=2) # returns a number of matches 

    return match 

 