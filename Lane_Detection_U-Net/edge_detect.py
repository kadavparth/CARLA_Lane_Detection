import numpy as np 
import cv2 

def binary_array(array : np.array([]), thresh : int, value = 0):

    if value == 0:
        binary = np.ones_like(array)
    else:
        binary = np.ones_like(array)
        value = 1

    binary[(array >= thresh[0]) & (array <= thresh[1])] = value

    return binary


def blur_gaussian(channel, kernel_size=3):

    return cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)


def mag_thresh(image, sobel_kernel=3, thresh=(0,255)):

    sobelx = np.abs(sobel(image, orient = 'x', sobel_kernel=sobel_kernel))

    sobely = np.abs(sobel(image, orient = 'y', sobel_kernel=sobel_kernel))

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    return binary_array(mag,thresh)


def sobel(img_channel, orient='x', sobel_kernel=3):

    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1,0 ,sobel_kernel)

    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0,1 ,sobel_kernel)

    return sobel 

def threshold(channel, thresh=(128,255), thresh_type = cv2.THRESH_BINARY):

    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


