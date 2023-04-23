import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import math

def extract_features(image):
    """
    Extract ORB descriptors from an image
    """

    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(image, None)

    # surf = cv2.SIFT_create(400) #(nfeatures=1500)
    # kp, des = surf.detectAndCompute(image, None)

    return kp, des


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
        kp, des = extract_features_function(image)
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
    index_params = dict(algorithm=FlANN_INDEX_KDTREE, trees=16)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match = flann.knnMatch(
        np.float32(des1), np.float32(des2), k=2
    )  # returns a number of matches

    return match


def filtered_match(mathces, dist_threshold=0.5):
    """
    match - list of all matches

    dist_threshold - maximum allowed relative distance between
    the best macthes (0.0, 1.0)

    """

    filtered_match = []

    for m, n in mathces:
        if m.distance / n.distance < dist_threshold:
            filtered_match.append([m, None])

    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, matches):
    """
    function to visualise mathces between two images
    takes in image1, kp1, image2, kp2, mathces list

    """

    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
    return image_matches


def estimate_motion(match, kp1, kp2, K, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
        
    ### START CODE HERE ###
    for m in match:
        m = m[0]
        query_idx = m.queryIdx
        train_idx = m.trainIdx

        # get first img matched keypoints
        p1_x, p1_y = kp1[query_idx].pt
        image2_points.append([p1_x, p1_y])

        # get second img matched keypoints
        p2_x, p2_y = kp2[train_idx].pt
        image1_points.append([p2_x, p2_y])

    # essential matrix
    E, mask = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), K)
    _, R, t, mask = cv2.recoverPose(E, np.array(image1_points), np.array(image2_points), K)
    
    rmat = R
    tvec = t  

    
    ### END CODE HERE ###
    
    return rmat, tvec, image1_points, image2_points

# def estimate_motion(match, kp1, kp2, k, depth1=None):
#     """
#     Estimate camera motion from a pair of subsequent image frames

#     retruns:
#     rmat, tvec, image1_pts, image2_pts
#     """

#     rmat = np.eye(3)
#     tvec = np.zeros((3, 1))
#     image1_pts = []
#     image2_pts = []

#     image2_pts = [kp2[m.trainIdx].pt for m,n in match]
#     image1_pts = [kp1[m.queryIdx].pt for m,n in match]

#     ### if we provide no depth

#     k = k.reshape(3,3)

#     if depth1 is None:
#         pts1 = np.array(image1_pts)
#         pts2 = np.array(image2_pts)

#         E, mask_match = cv2.findEssentialMat(pts1, pts2, k, method=cv2.RANSAC)

#         _, rmat, tvec, _ = cv2.recoverPose(E, pts1, pts2, k)

#     ### if we provide depth

#     else:
#         f, cu, cv = k[0, 0], k[0, 2], k[1, 2]

#         objectPoints = cv2.convertPointsToHomogeneous(np.array(image1_pts))

#         i = 0

#         for x, y in image1_pts:
#             z = depth1[int(y)][int(x)]
#             objectPoints[:, :, 0][i] = z * (x - cu) / f
#             objectPoints[:, :, 1][i] = z * (y - cv) / f
#             objectPoints[:, :, 2][i] = z

#             i += 1

#         retval, rvec, tvec = cv2.solvePnP(
#             np.array(objectPoints),
#             np.array(image2_pts),
#             k,
#             None,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#         )

#         rmat, _ = cv2.Rodrigues(rvec)

#     return rmat, tvec, image1_pts, image2_pts


def match_features_dataset(des_list, match_features):
    matches = []

    for i in range(len(des_list) - 1):
        match = match_features(des_list[i], des_list[i + 1])

        matches.append(match)

    return matches


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 5, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1


def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    
    ### START CODE HERE ###
    for m,n in matches:
        if m.distance/n.distance < dist_threshold:
            filtered_match.append([m,None])
    
    ### END CODE HERE ###

    return filtered_match


def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = []
    
    ### START CODE HERE ###
    for i in range(len(matches)):
        match = filter_matches_distance(matches[i], dist_threshold)
        filtered_matches.append(match)

    
    ### END CODE HERE ###
    
    return filtered_matches


def estimate_trajectory(estimate_motion, matches, kp_list, K, depth_maps=[]):

    trajectory = np.zeros((3,1))
    
    trajectory = [np.array([0,0,0])]
    P = np.eye(4)

    # R = np.diag([1,1,1])
    # T = np.zeros([3,1])
    # RT = np.hstack([R,T])
    # RT = np.vstack([RT, np.zeros([1,4])])
    # RT[-1, -1] = 1

    for i in range(len(matches)):

        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth = depth_maps[i]

        rmat, tvec, _ , _ = estimate_motion(match, kp1, kp2, K, depth1=depth)
        # rt_mtx = np.hstack([rmat, tvec])
        # rt_mtx = np.vstack([rt_mtx, np.zeros([1,4])])
        # rt_mtx[-1,1] = 1 

        R = rmat
        t = np.array([tvec[0,0], tvec[1,0], tvec[2,0]])

        P_new = np.eye(4)
        P_new[0:3, 0:3] = R.T
        P_new[0:3, 3] = (-R.T).dot(t)
        P = P.dot(P_new)

        trajectory.append(P[:3,3])

        # rt_mtx_inv = np.linalg.inv(rt_mtx)

        # RT = RT @ rt_mtx_inv

        # new_trajectory = RT[:3,3]
        # trajectory.append(new_trajectory)

    trajectory = np.array(trajectory).T
    trajectory[2,:] = -1*trajectory[2,:]

    return trajectory




def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]
        
        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)
    
    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()
    plt.show()