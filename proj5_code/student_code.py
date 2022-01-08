import numpy as np
import math
import os

import glob
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt

from colour import Color
from scipy.spatial.transform import Rotation

# TODO 1
def calculate_projection_matrix(points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    #world = points_3d
    #image = points_2d
    #B = []
    #for i in np.arange(image.shape[0]):
    #    eq1 = np.array([world[i,0],world[i,1],world[i,2],1,0,0,0,0, -image[i,0]*world[i,0], -image[i,0]*world[i,1],-image[i,0]*world[i,2], -image[i,0]])
     #   eq2 = np.array([0,0,0,0,world[i,0],world[i,1],world[i,2],1, -image[i,1]*world[i,0], -image[i,1]*world[i,1],-image[i,1]*world[i,2], -image[i,1]])
      #  A = np.vstack((eq1,eq2))
      #  B.append(A)
      #  C = np.asarray(B)
    #u,s,v = np.linalg.svd(C)
    #y = v.shape[0]
    #X = v[y-1,:]
    #z = X.shape[0]
    #X = X[z-1,:]
    #M = X.reshape(3,4)
    #--------------------------
    a_mat = np.zeros([len(points_3d) * 2 , 11])
    b_mat = np.zeros([len(points_3d) * 2, 1])
    
    for index in range(len(points_2d)):
        X1, Y1, Z1 = points_3d[index]
        u1, v1 = points_2d[index]
        
        a_mat[index * 2] = np.array([X1, Y1, Z1, 1, 0,  0,  0, 0, -u1 * X1, -u1 * Y1, -u1 * Z1])
        a_mat[index * 2 + 1] = np.array([0,  0,  0,  0, X1, Y1, Z1, 1, -v1 * X1, -v1 * Y1, -v1 * Z1])
        b_mat[index * 2] = u1
        b_mat[index * 2 + 1] = v1
    
    M, residuals, rank, s = np.linalg.lstsq(a_mat, b_mat, rcond = None)
    M = np.vstack([M, 1]).reshape(3, 4)
    #raise NotImplementedError('`calculate_projection_matrix` function in ' +
    #'`student_code.py` needs to be implemented')

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return M

# TODO 2
def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
    -  P: 3x4 projection matrix
    -  points_3d : n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
    - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    #----------------------->>>>if(points_3d.shape[1] == 3):<<<<<
    points_3d = np.append(points_3d, np.ones((points_3d.shape[0],1)), axis = 1)

    empty = np.dot(P, points_3d.T)

    projected_points_2d = np.zeros((len(points_3d), 2))
    projected_points_2d[:, 0] = (empty[0] / empty[2]).T
    projected_points_2d[:, 1] = (empty[1] / empty[2]).T
    #raise NotImplementedError('`projection` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return projected_points_2d

# TODO 3
def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    Q = M[:, :3]
    m4 = M[:, 3]
    cc = -np.linalg.inv(Q).dot(m4)
    #raise NotImplementedError('`calculate_camera_center` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
    return cc

# TODO 4
def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
    -   points: A numpy array of shape (N, 2) representing the 2D points in
                the image

    Returns:
    -   points_normalized: A numpy array of shape (N, 2) representing the normalized 2D points in
                the image
    -   T: Transform matrix representing the product of the scale and offset matrices
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    cu, cv = np.mean(points, axis = 0)
    denum_su = np.std((points - np.array([cu, cv]))[:,0])
    denum_sv = np.std((points - np.array([cu, cv]))[:,1])
    su = 1.0 / denum_su
    sv = 1.0 / denum_sv

    s_matrix = np.array([[su, 0 ,0],[0,sv, 0],[0,0,1]])
    o_matrix = np.array([[1, 0, -cu],[0, 1, -cv],[0, 0, 1]])
    T = s_matrix.dot(o_matrix)

    points_ones = np.ones([len(points),3])
    points_ones[:,0:2] = points

    points_normalized_ones = points_ones.dot(T.T) #(N,3).dot(3,3) = (N,3)
    points_normalized = points_normalized_ones[:,0:2]
    #raise NotImplementedError('`normalize_points` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return points_normalized, T

# TODO 5
def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjust F to account for normalized coordinates by using the transform matrices.
    Args:
    -   F_norm: A numpy array of shape (3, 3) representing the normalized fundamental matrix
    -   T_a: Transform matrix for image A
    -   T_B: Transform matrix for image B

    Returns:
    -   F_orig: A numpy array of shape (3, 3) representing the original fundamental matrix
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    F_orig = T_b.T.dot(F_norm).dot(T_a)
    #raise NotImplementedError('`unnormalize_F` function in ' +
    #'`student_code.py` needs to be implemented')

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return F_orig

# TODO 6
def estimate_fundamental_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculates the fundamental matrix.
    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    A = np.zeros([len(points_a), 8]) 
    normal_a, T_a = normalize_points(points_a)
    normal_b, T_b = normalize_points(points_b)
    
    for i in range(normal_a.shape[0]):
        ua, va = normal_a[i]
        ub, vb = normal_b[i]
        A[i] = [ua * ub, va * ub, ub, ua * vb, va * vb, vb, ua, va]

    local = -1 * np.ones(A.shape[0])
    F, residuals, rank, s = np.linalg.lstsq(A, local, rcond = None)
    F = np.append(F,1).reshape(3,3)
    
    U, S, Vh = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    
    F_norm = U.dot(S).dot(Vh)
    
    F = unnormalize_F(F_norm, T_a, T_b)

    #raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
    #'`student_code.py` needs to be implemented')

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return F

# TODO 7
def compute_fundamental_matrix_ransac(img1_path, img2_path, img1_scale, img2_scale, num_features):
    """
    Estimate the fundamental matrix both with and without RANSAC. 

    Tips:
        1. Use the helper functions load_image to load the images after passing the image paths
        2. Use a function in OpenCV such as cv2.resize to resize the images with appropriate scale factors
        3. Use the function get_matches to get point correspondences between the image pairs
        4. To estimate the fundamental matrices, use the functions defined/coded in earlier parts
        5. Use the show_correspondence2 to generate the visualizations of image correspondences with/without RANSAC
           You need to pass the appropriate matched points to get different outputs with/without RANSAC

    Args:
    -   img1_path: Path to the first image
    -   img2_path: Path to the second image
    -   img1_scale: Scaling factor for image 1
    -   img2_scale: Scaling factor for image 2
    -   num_features: an int representing number of matching points required.

    Returns:
    -   *match_image: Image correspondences without RANSAC*
    -   match_image: Image correspondence visualization without RANSAC
    -   match_image_ransac: Image correspondence visualization with RANSAC
    -   matched_points_wo_ransac_a: Matched points without RANSAC in Image 1 *****
    -   matched_points_wo_ransac_b: Matched points without RANSAC in Image 2 *****
    -   matched_points_ransac_a: Matched points with RANSAC in Image 1 *****
    -   matched_points_ransac_b: Matched points with RANSAC in Image 2 *****
    -   pic_a: Image 1 loaded using the load_image helper function *****
    -   pic_b: Image 2 loaded using the load_image helper function *****
    -   F_wo_ransac: Fundamental matrix estimated without using RANSAC *****
    -   F_ransac: Fundamental matrix estimated using RANSAC *****   
    """
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    from proj5_code.utils import load_image
    pic_a = load_image(img1_path)
    pic_b = load_image(img2_path)
    pic_a = cv2.resize(pic_a,(0,0), fx = img1_scale, fy = img1_scale)
    pic_b = cv2.resize(pic_b,(0,0), fx = img2_scale, fy = img2_scale)
    image1 = pic_a
    image2 = pic_b
    
    matched_points_wo_ransac_a, matched_points_wo_ransac_b = get_matches(image1, image2, num_features)
    
    F_wo_ransac = estimate_fundamental_matrix(matched_points_wo_ransac_a, matched_points_wo_ransac_b)
    F_ransac, matched_points_ransac_a, matched_points_ransac_b = ransac_fundamental_matrix(matched_points_wo_ransac_a, matched_points_wo_ransac_b)
    
    X1 = matched_points_wo_ransac_a[:,0]
    Y1 = matched_points_wo_ransac_a[:,-1]
    X2 = matched_points_wo_ransac_b[:,0]
    Y2 = matched_points_wo_ransac_b[:,-1]
    match_image = show_correspondence2(pic_a, pic_b, X1, Y1, X2, Y2)
    
    x1 = matched_points_ransac_a[:,0]
    y1 = matched_points_ransac_a[:,-1]
    x2 = matched_points_ransac_b[:,0]
    y2 = matched_points_ransac_b[:,-1]
    match_image_ransac = show_correspondence2(pic_a, pic_b, x1, y1, x2, y2)
    
    #raise NotImplementedError('`compute_fundamental_matrix_ransac` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
    
    return match_image, match_image_ransac, matched_points_wo_ransac_a, matched_points_wo_ransac_b, matched_points_ransac_a, matched_points_ransac_b, pic_a, pic_b, F_wo_ransac, F_ransac

# TODO 8
def get_visual_odometry(images_path) -> List[np.ndarray]:
    """ 
    Images are loaded from dataset and relative poses are calculated.
    
    Your task is to create a list "img_fpaths" that holds the paths to image frames located in "images_path".
    Make sure that the paths of image frames are in proper(sorted) order. Rest of the code for visual_odometry is already written. 
    
    
    Args : 
        - images_path : path to directory containing sequential image frames
        
    Returns : 
        - iCurrTiPrev : numpy array of relative poses between current and previous frames.
        
    """
    img_fpaths = []
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    img_wildcard = f"{DATA_ROOT}/vo_seq_argoverse_273c1883/ring_front_center/*.jpg"
    img_fpaths = glob.glob(img_wildcard)
    
    img_fpaths.sort()
    num_imgs = len(img_fpaths)
    print(num_imgs)
    K = load_log_front_center_intrinsics()
    
    iCurrTiPrev = []
    #iCurrTiPrev += [np.eye(4)]

    for i in range(num_imgs - 1):
        img_i1 = load_image(img_fpaths[i])
        img_i2 = load_image(img_fpaths[i + 1])
        pts_a, pts_b = get_matches(img_i1, img_i2, n_feat=int(4e3))

        # between camera at t=i and t=i+1
        i2_F_i1, inliers_a, inliers_b = ransac_fundamental_matrix(pts_a, pts_b)
        i2_E_i1 = get_emat_from_fmat(i2_F_i1, K1=K, K2=K)
        _num_inlier, i2Ri1, i2ti1, _ = cv2.recoverPose(i2_E_i1, inliers_a, inliers_b)
        
        # form SE(3) transformation
        i2Ti1 = np.eye(4)
        i2Ti1[:3, :3] = i2Ri1
        i2Ti1[:3, 3] = i2ti1.squeeze()

        iCurrTiPrev += [i2Ti1];
        #iCurrTiPrev = np.asarray(iCurrTiPrev)
        
        r = Rotation.from_matrix(i2Ri1.T)
        rz, ry, rx = r.as_euler("zyx", degrees=True)
        print(f"Rotation about y-axis from frame {i} -> {i+1}: {ry:.2f} degrees")
        
    #raise NotImplementedError('`get_visual_odometry` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return iCurrTiPrev

# TODO 9
def compute_absolute_poses(iCurrTiPrev : List[np.ndarray])-> List[np.ndarray]:
    """
    Calculate absolute poses(world frame poses) from the relative poses. 
    
    Tips : Absolute pose can be calculated by performing a simple matrix multiplication of "previous world frame pose" 
    and the "relative pose between previous and current frames". 
    For instance, wTi(2) can be calculated using wTi(1) and i1Ti2  [ wTi(2) = wTi1*i1Ti2 ]
    Append each absolute wTi(n) to poses_wTi. 
    
    Note that the relative poses in iCurrTiPrev array are of the form [ [i2Ti1], [i3Ti2], [i4Ti3], ... ]. You will have
    to modify each relative pose accordingly(perform inverse operation) before performing necessary matrix multiplication to retrieve each absolute pose.
    
    
    Args : 
        - iCurrTiPrev : list of numpy arrays containing relative poses calculated using subsequent frames (current 'n', previous 'n-1')
    
    Returns : 
        - poses_wTi : list of numpy arrays containing absolute poses (world frame poses) of the form [[wTi(1)], [wTi(2)], [wTi(3)], ... ]
    """
    
    poses_wTi = []
    poses_wTi += [np.eye(4)] # initial pose wTi(1) is set to identity matrix. 
    
    
    #######################################################################
    # TODO: YOUR CODE HERE                                                #
    #######################################################################
    a = 0
    for index in range(1, len(iCurrTiPrev)+1):
        prev = iCurrTiPrev[index-1]
        inv_value = np.linalg.inv(prev)
        val = np.matmul(poses_wTi[index - 1], inv_value)
        print(a+1)
        a += 1
        poses_wTi += [val]
    
    poses_wTi = np.asarray(poses_wTi)
      
    #raise NotImplementedError('`compute_absolute_poses` function in ' +
    #'`student_code.py` needs to be implemented')
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
    
    return poses_wTi;

####################################### DO NOT MODIFY THE CODES BELOW #########################################################

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

def get_emat_from_fmat(i2_F_i1: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """ Create essential matrix from camera instrinsics and fundamental matrix"""
    i2_E_i1 = K2.T @ i2_F_i1 @ K1
    return i2_E_i1


def load_log_front_center_intrinsics() -> np.array:
    """Provide camera parameters for front-center camera for Argoverse vehicle log ID:
    273c1883-673a-36bf-b124-88311b1a80be
    """
    fx = 1392.1069298937407  # also fy
    px = 980.1759848618066
    py = 604.3534182680304

    K = np.array([[fx, 0, px], [0, fx, py], [0, 0, 1]])
    return K

def plot_poses(poses_wTi: List[np.ndarray], figsize=(7, 8)) -> None:
    """
    Poses are wTi (in world frame, which is defined as 0th camera frame)
    """
    axis_length = 0.5

    num_poses = len(poses_wTi)
    colors_arr = np.array([[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), num_poses)]).squeeze()

    _, ax = plt.subplots(figsize=figsize)

    for i, wTi in enumerate(poses_wTi):
        wti = wTi[:3, 3]

        # assume ground plane is xz plane in camera coordinate frame
        # 3d points in +x and +z axis directions, in homogeneous coordinates
        posx = wTi @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
        posz = wTi @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

        ax.plot([wti[0], posx[0]], [wti[2], posx[2]], "b", zorder=1)
        ax.plot([wti[0], posz[0]], [wti[2], posz[2]], "k", zorder=1)

        ax.scatter(wti[0], wti[2], 40, marker=".", color=colors_arr[i], zorder=2)

    plt.axis("equal")
    plt.title("Egovehicle trajectory")
    plt.xlabel("x camera coordinate (of camera frame 0)")
    plt.ylabel("z camera coordinate (of camera frame 0)")

def load_image(path: str) -> np.ndarray:
    return cv2.imread(path)[:, :, ::-1]


def get_matches(pic_a: np.ndarray, pic_b: np.ndarray, n_feat: int) -> (np.ndarray, np.ndarray):
    """Get unreliable matching points between two images using SIFT.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        pic_a: a numpy array representing image 1.
        pic_b: a numpy array representing image 2.
        n_feat: an int representing number of matching points required.

    Returns:
        pts_a: a numpy array representing image 1 points.
        pts_b: a numpy array representing image 2 points.
    """
    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp_a, desc_a = sift.detectAndCompute(pic_a, None)
    kp_b, desc_b = sift.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_L2)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance / 1.2:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches[: int(n_feat)]:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)

def hstack_images(imgA: np.ndarray, imgB: np.ndarray) -> np.ndarray:
    """Stacks 2 images side-by-side

    Args:
        imgA: a numpy array representing image 1.
        imgB: a numpy array representing image 2.

    Returns:
        img: a numpy array representing the images stacked side by side.
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[: imgA.shape[0], : imgA.shape[1], :] = imgA
    newImg[: imgB.shape[0], imgA.shape[1] :, :] = imgB

    return newImg


def show_correspondence2(
    imgA: np.ndarray, imgB: np.ndarray, X1: np.ndarray, Y1: np.ndarray, X2: np.ndarray, Y2: np.ndarray, line_colors=None
) -> None:
    """Visualizes corresponding points between two images. Corresponding points
    will have the same random color.

    Args:
        imgA: a numpy array representing image 1.
        imgB: a numpy array representing image 2.
        X1: a numpy array representing x coordinates of points from image 1.
        Y1: a numpy array representing y coordinates of points from image 1.
        X2: a numpy array representing x coordinates of points from image 2.
        Y2: a numpy array representing y coordinates of points from image 2.
        line_colors: a N x 3 numpy array containing colors of correspondence
            lines (optional)

    Returns:
        None
    """
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    dot_colors = np.random.rand(len(X1), 3)
    if imgA.dtype == np.uint8:
        dot_colors *= 255
    if line_colors is None:
        line_colors = dot_colors

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors, line_colors):
        newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        newImg = cv2.circle(newImg, (x2 + shiftX, y2), 5, dot_color, -1)
        newImg = cv2.line(newImg, (x1, y1), (x2 + shiftX, y2), line_color, 2, cv2.LINE_AA)

    return newImg

def calculate_num_ransac_iterations(prob_success: float, sample_size: int, ind_prob_correct: int) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ##############################
    numerator = math.log10(1.0 - prob_success)
    denominator = math.log10(1.0 - (ind_prob_correct ** sample_size))
    num_samples = numerator / denominator
    ##############################

    return int(num_samples)


def ransac_fundamental_matrix(matches_a: np.ndarray, matches_b: np.ndarray) -> np.ndarray:
    """
    For this section, we use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. 

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    best_F = np.empty((3, 4), dtype=np.float)
    best_inliers_count = 0
    inliers_a = np.empty((0, 0))
    inliers_b = np.empty((0, 0))

    N = matches_a.shape[0]
    sample_size = 8
    threshold = 0.1

    inlier_fraction = 0.5
    desired_success_probability = 0.99
    max_iterations = calculate_num_ransac_iterations(desired_success_probability, sample_size, inlier_fraction)

    X_a = np.hstack((matches_a, np.ones((N, 1))))
    X_b = np.hstack((matches_b, np.ones((N, 1))))

    for _ in range(max_iterations):
        sample_indices = np.random.randint(0, high=N, size=sample_size)
        sample_points_a = matches_a[sample_indices, :]
        sample_points_b = matches_b[sample_indices, :]
        F = estimate_fundamental_matrix(sample_points_a.copy(), sample_points_b.copy())
        inlier_indices = []
        for ind in range(N):
            xai = X_a[ind, :]
            xbi = X_b[ind, :]
            line = np.dot(F, np.transpose(xai))
            line = line / np.sqrt(line[0] ** 2 + line[1] ** 2)  # normalizes the line vector
            distance = np.abs(np.dot(line, np.transpose(xbi)))
            if distance < threshold:
                inlier_indices.append(ind)
        if len(inlier_indices) > best_inliers_count:
            best_F = F
            best_inliers_count = len(inlier_indices)
            inliers_a = matches_a[inlier_indices, :]
            inliers_b = matches_b[inlier_indices, :]

    return best_F, inliers_a, inliers_b