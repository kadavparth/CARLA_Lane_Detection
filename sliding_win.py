import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def Perspective(img):
    
    # src = np.float32([[0,745], [150,481], [580,481], [800,745]]) # works well
    src = np.float32([[0,800], [90,481], [630,481], [800,800]])
    dst = np.float32([[0,img.shape[0]], [0,0], [img.shape[1], 0], [img.shape[1], img.shape[0]]])
    
    M = cv2.getPerspectiveTransform(src,dst)
    
    Minv = cv2.getPerspectiveTransform(dst,src)
    
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, Minv, unwarped

def histogram_peak(histogram):
    
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
 
    return leftx_base, rightx_base

def calc_hist(frame):
    
    histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
    
    return histogram 


def get_lane_line_indices_sliding_window(warped_frame,histogram):
    
    frame_sliding_window = warped_frame.copy()
    nwindows = 12
    margin = 65
    minpix = 50
    window_height = int(warped_frame.shape[0]/nwindows)
    
    nonzero = warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
    
    left_lane_inds = []
    right_lane_inds = []
    
    leftx_base, rightx_base = histogram_peak(histogram)
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    for window in range(nwindows):
        
      win_y_low = warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
        win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
        win_xright_high,win_y_high), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      minpix = minpix
      
      if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2) 
    
    ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    midx = (left_fitx + right_fitx)/2
    
    out_img = np.dstack((frame_sliding_window, frame_sliding_window, 
                         (frame_sliding_window))) * 255

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, left_fitx, right_fitx, midx, ploty ,out_img, frame_sliding_window


# def LanePixels(binary_warped, plot=False):
    
#     histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:,:],axis=0)
    
#     if plot==True:
#         plt.plot(histogram)
#         print(histogram.shape)
        
#     out_img = np.dstack((binary_warped,binary_warped,binary_warped))
    
#     midpoint = np.int64(histogram.shape[0] // 2)
#     leftx_base = np.argmax(histogram[:midpoint])
#     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
#     nwindows = 12
    
#     margin = 65
    
#     minpix = 50
    
#     window_height = np.int64(binary_warped.shape[0] // nwindows)
    
#     nonzero = binary_warped.nonzero()
#     nonzerox = np.array(nonzero[1])
#     nonzeroy = np.array(nonzero[0])
    
#     leftx_current = leftx_base 
#     rightx_current = rightx_base 
    
#     left_lane_inds = []
#     right_lane_inds = []
    
#     for window in range(nwindows):
        
#         win_y_low = binary_warped.shape[0] - (window + 1) * window_height
#         win_y_high = binary_warped.shape[0] - window * window_height
#         win_xleft_low = leftx_current - margin
#         win_xleft_high = leftx_current + margin 
#         win_xright_low = rightx_current - margin 
#         win_xright_high = rightx_current + margin 
        
#         # Draw the rectangles on the image 
        
#         cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0),2)
#         cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0),2)
        
#         good_left_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) 
#                          & (nonzerox < win_xleft_high)).nonzero()[0]
#         good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) 
#                          & (nonzerox < win_xright_high)).nonzero()[0]
        
#         left_lane_inds.append(good_left_inds)
#         right_lane_inds.append(good_right_inds)
        
#         if len(good_left_inds) > minpix:
#             leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
#         if len(good_right_inds) > minpix:
#             rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
            
            
#     try:
#         left_lane_inds = np.concatenate(left_lane_inds)
#         right_lane_inds = np.concatenate(right_lane_inds)
        
#     except ValueError:
#         pass 
    
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
    
#     return leftx, lefty, rightx, righty, out_img


# def FitPoynomial(binary_warped, plot = False):
    
#     leftx, lefty, rightx, righty, out_img = LanePixels(binary_warped)
    
#     left_fit = np.polyfit(leftx, lefty, 2)
#     right_fit = np.polyfit(rightx, righty, 2)

#     middle_fit = (left_fit + right_fit) / 2
    
#     ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

#     # blank_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1],3),dtype=np.uint8)

#     try: 
#         left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] + left_fit[2]
#         middle_fitx = middle_fit[0] * plot ** 2 + middle_fit[1] + middle_fit[2]
#         right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] + right_fit[2]
    
#     except TypeError: 
        
#         print('the function failed to fit a line')
#         left_fitx = 1 * ploty ** 2 + 1 * ploty
#         middle_fitx = 1* plot ** 2 + 1 * ploty
#         right_fitx = 1 * ploty ** 2 + 1 * ploty
    
#     if plot == True:

#         out_img[lefty, leftx] = [255,0,0]
#         out_img[righty, rightx] = [0,0,255]

#     return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, middle_fit


# def get_warp_lines(warped_img, pts_left, pts_right, pts_mid, Minv):

#     lwarped_img = np.copy(warped_img)
#     rwarped_img = np.copy(warped_img)
#     mwarped_img = np.copy(warped_img)

#     # left lane Lines
#     image = cv2.polylines(warped_img, [pts_left], False, 
#                     color= (255,255,255), thickness=2)

#     newwarp = cv2.warpPerspective(image, Minv, (800,800))

#     idxs_left = np.where( ( newwarp[:,:,0] == 255) & (newwarp[:, :,1] == 255) & (newwarp[:, :, 2] == 255)  )

#     # right lane Lines
#     image = cv2.polylines(warped_img, [pts_right], False, 
#                     color= (255,255,255), thickness=2)

#     newwarp = cv2.warpPerspective(image, Minv, (800,800))

#     idxs_right = np.where( ( newwarp[:,:,0] == 255) & (newwarp[:, :,1] == 255) & (newwarp[:, :, 2] == 255)  )

#     # middle lane Lines
#     image = cv2.polylines(warped_img, [pts_mid], False, 
#                     color= (255,255,255), thickness=2)

#     newwarp = cv2.warpPerspective(image, Minv, (800,800))

#     idxs_mid = np.where( ( newwarp[:,:,0] == 255) & (newwarp[:, :,1] == 255) & (newwarp[:, :, 2] == 255)  )

#     return idxs_left, idxs_right, idxs_mid, image

