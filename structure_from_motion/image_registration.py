import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import re

class ImageRegistration:
    """
    A class to perform perform image mosiacking operations on the Skerski dataset using GTSAM.
    """

    def __init__(self, image_directories,
                 visualize_features = False,
                 visualize_epipolar_lines = False,
                 sift_nOctaveLayers = 3,
                 sift_contrastThreshold = 0.04,
                 sift_edgeThreshold = 10,
                 sift_sigma = 1.6,
                 ransac_reproj_thres = 5.0):
        """
        Initializes the ImageRegistration class with placeholder values.
        """
        self.images = []

        # Image directory
        self.image_directories = image_directories

        # SIFT
        self.sift = cv2.SIFT.create(nOctaveLayers = sift_nOctaveLayers,
                                    contrastThreshold=sift_contrastThreshold,
                                    edgeThreshold = sift_edgeThreshold,
                                    sigma = sift_sigma)


        self.focal_length = 2000
        self.visualize_features = visualize_features
        self.visualize_epipolar_lines = visualize_epipolar_lines
        self.ransac_reproj_thres = ransac_reproj_thres

    def set_camera_intrinsics(self, f, px, py):
        self.camera_intrinsics = np.array([[f, 0, px],
                          [0, f, py],
                          [0, 0, 1]])

    def get_images_for_registration(self):
        """
        Loads all images from multiple subdirectories for mosaicking and displays them.
        Images are read in numerical order based on their filenames.

        Sets the camera intrinsics.
        """

        # Function to extract numbers from filenames for numerical sorting
        def extract_number(filename):
            # Find the first number in the filename
            match = re.search(r'\d+', filename)
            # Return inf if no number is found
            return int(match.group()) if match else float('inf')

        # Initialize images list
        self.images = []

        # Iterate through each subdirectory in the list
        for directory in self.image_directories:
            # Get the current working directory
            current_dir = os.getcwd()

            # Create the full path to the subdirectory
            full_path = os.path.join(current_dir, directory)

            # Check if the subdirectory exists
            if not os.path.exists(full_path):
                print(f"Warning: Directory {full_path} does not exist.")
                continue

            # Get and sort image files numerically
            image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            image_files = sorted(image_files, key=extract_number)

            # Loop through all sorted image files and load them
            for filename in image_files:
                file_path = os.path.join(full_path, filename)

                # Load the image using OpenCV
                image = cv2.imread(file_path)

                # Append the image if it's loaded correctly
                if image is not None:
                    self.images.append(image)
                else:
                    print(f"Warning: Failed to load {filename}")

            # Determine the grid size for displaying images
            num_images = len(self.images)
            # Number of columns in the grid.
            cols = 4
            rows = math.ceil(num_images / cols)

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop through the images and plot each in the corresponding grid cell
        for i, image in enumerate(self.images):
            # Convert BGR to RGB for display in Matplotlib
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[i].axis('off')

        # Turn off any unused axes (if the grid is larger than the number of images)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()

        # Get image height and width
        num_rows, num_cols = self.images[0].shape[:2]

        # Set camera intrinsics.
        self.set_camera_intrinsics(self.focal_length, num_cols/2, num_rows/2)

    def get_features(self):
        """
        Detects SIFT keypoints and descriptors for each image in `self.images`, stores them in
        `self.keypoints` and `self.descriptors`, and visualizes the keypoints on the grayscale images
        in a grid layout.

        Keypoints and descriptors are extracted from each image after conversion to grayscale for
        efficiency. The detected keypoints are drawn on the images and displayed in a grid using Matplotlib.

        Attributes:
            self.keypoints (list): Keypoints detected in each image.
            self.descriptors (list): Corresponding descriptors for the keypoints.

        """
        self.keypoints = []
        self.descriptors = []
        self.object_index_list = []

        # Set up the grid layout for plotting
        num_images = len(self.images)
        # Number of columns in the grid.
        num_cols = 4
        # Calculate number of rows needed
        num_rows = (num_images + num_cols - 1) // num_cols

        if self.visualize_features:
            # Create a Matplotlib figure
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
            # Flatten the grid into a 1D array for easy access
            axes = axes.ravel()

        for i, image in enumerate(self.images):

            # Convert image to grayscale to reduce the computational load on SIFT by reducing
            # intensity to a single channel.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get keypoints and descriptors.
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)

            # Create object index list for the image.
            # NOTE: Refered from ZZ's code.
            self.object_index_list.append(np.full(len(keypoints), -1, int))

            # Draw the keypoints on the grayscale image
            image_with_keypoints = cv2.drawKeypoints(
                gray_image, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if self.visualize_features:
                # Plot the image with keypoints in the grid
                axes[i].imshow(image_with_keypoints, cmap='gray')
                axes[i].set_title(f'Image {i+1} with Keypoints')
                axes[i].axis('off')

        if self.visualize_features:
            # Hide any remaining empty subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')

            # Show the grid of images
            plt.tight_layout()
            plt.show()

    def match_features(self, index1, index2):
        """
        Finds good matches from the keypoints and descripters.

        Utilizes a brute force approach to find potential matches and
        reject matches if the second best match is too close.

        Args:
            index1 (int): Index of the first image from `self.images`.
            index2 (int): Index of the second image from `self.images`.

        Returns:
            good_matches (list of cv2.DMatch): List of good matches between the two images.
        """
        brute_force_matcher = cv2.BFMatcher()

        # Get matches between the two images, alongwith a second best match for each batch.
        matches = brute_force_matcher.knnMatch(self.descriptors[index1], self.descriptors[index2], k=2)

        good_matches = []
        index1_keypoint_index_list = []
        index2_keypoint_index_list = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good_matches.append([m])
                # Store the indices of the points in good matches
                index1_keypoint_index_list.append(m.queryIdx)
                index2_keypoint_index_list.append(m.trainIdx)

        return good_matches, np.array(index1_keypoint_index_list), np.array(index2_keypoint_index_list)


    # TODO(KSorte): Implement an iterative 8 point algorithm to compute fundamental matrix.
    def compute_fundamental_matrix(self, first_index, second_index, good_matches, first_indices, second_indices):
        """
        Computes the fundamental matrix between two images using matched keypoints.

        The function takes matched keypoints between two images, and applies RANSAC
        to compute the fundamental matrix.

        Args:
            first_index (int): Index of the source image in `self.images`.
            second_index (int): Index of the destination image in `self.images`.
            good_matches (list of cv2.DMatch): List of good matches between the two images.

        Returns:
            numpy.ndarray: The computed fundamental matrix (3x3).
        """
        # Get coordinates of the matched keypoints from both images.
        first_pts = np.float32([self.keypoints[first_index][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        second_pts = np.float32([self.keypoints[second_index][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        F, mask = cv2.findFundamentalMat(first_pts, second_pts, cv2.FM_RANSAC, ransacReprojThreshold=self.ransac_reproj_thres)
        # Select only inlier points
        inlier_points1 = first_pts[mask.ravel()==1]
        inlier_points2 = second_pts[mask.ravel()==1]

        first_indices = first_indices[mask.ravel()==1]
        second_indices = second_indices[mask.ravel()==1]

        # Separate good matches into inliers and outliers based on the mask.
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

        # TODO(KSorte): Handle this return tuple better.
        return F, inlier_points1, inlier_points2, inlier_matches, first_indices, second_indices

    # TODO(KSorte): Split this function up.
    def process_epipolar_geometry_and_recover_relative_poses(self):
        """
        Computes feature matches, fundamental matrices, essential matrices, and relative poses between consecutive images in a sequence.

        Matches keypoints between consecutive images, visualizes the matches, computes the fundamental
        matrix for each image pair using RANSAC, computes the essential matrix, and retrieves the relative pose.

        Attributes:
            self.image_feature_matches (list): Stores lists of good matches between consecutive images.
            self.fundamental_matrices (list): Stores the fundamental matrices computed between consecutive image pairs.
            self.essential_matrices (list): Stores the essential matrices computed between consecutive image pairs.
            self.relative_poses (list): Stores the relative poses (rotation and translation) for consecutive image pairs.
        """

        self.essential_matrices = []
        self.relative_poses = []
        self.inlier_points = []
        self.inlier_indices = []

        for i in range(len(self.images) - 1):
            # Get matches for i and i + 1 th image
            good_matches, first_indices, second_indices = self.match_features(i, i+1)

            # Get Fundamental Matrix and inlier matches.
            F, inlier_points1, inlier_points2, inlier_matches, first_indices, second_indices = \
                self.compute_fundamental_matrix(i, i+1, good_matches, first_indices, second_indices)

            # TODO(KSorte): Add a condition that min 5 inliers needed.
            # Compute essential matrix.
            E = ImageRegistration.get_essential_matrix(F, self.camera_intrinsics)

            # Recover relative pose.
            _, R, T, pose_mask = cv2.recoverPose(E, inlier_points1, inlier_points2, self.camera_intrinsics)
            self.relative_poses.append((R, T))

            # Store E
            self.essential_matrices.append(E)

            inlier_points1 = inlier_points1[pose_mask == 255]
            inlier_points2 = inlier_points2[pose_mask == 255]
            pose_mask = np.squeeze(pose_mask != 0)

            first_indices = first_indices[pose_mask]
            second_indices = second_indices[pose_mask]

            self.inlier_indices.append((first_indices, second_indices))

            if inlier_points1.shape[0] == 0 or inlier_points2.shape[0] == 0:
                print("empty inliers")

            self.inlier_points.append((inlier_points1, inlier_points2))

            if self.visualize_epipolar_lines:
                # Compute and Visualize epipolar lines
                self.compute_and_draw_epipolar_lines(i, i+1, inlier_matches, F)


    def compute_camera_extrinsics(self):
        """
        Computes absolute poses (4x4 transformation matrices) for each image in the sequence
        from relative poses (stored as tuples) between consecutive images.

        Assumes the first image is at the origin with an identity pose.

        Attributes:
            self.camera_extrinsic_poses (list): Stores the 4x4 transformation matrices for each image in the sequence.
        """
        # Initialize the absolute poses list
        self.camera_extrinsic_poses = []

        # Start with the first pose as the identity matrix (4x4)
        initial_pose = np.eye(4)
        self.camera_extrinsic_poses.append(initial_pose)

        # Iterate through relative poses to compute the absolute poses
        for i, (relative_rotation, relative_translation) in enumerate(self.relative_poses):
            # Retrieve the previous absolute pose
            prev_pose = self.camera_extrinsic_poses[i]

            # Construct the 4x4 transformation matrix for the current relative pose
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = relative_rotation
            relative_pose[:3, 3] = relative_translation.flatten()

            # Compute the new absolute pose by chaining the previous absolute pose and the relative pose
            current_pose = relative_pose @ prev_pose

            # Store the absolute pose for the current image
            self.camera_extrinsic_poses.append(current_pose)

    def triangulate_landmarks_all_views(self):
        self.world_points_3D = []
        for i in range(len(self.images) - 1):
            # Inlier points
            points_1 = self.inlier_points[i][0]
            points_2 = self.inlier_points[i][1]

            # Homogeneous 4D world points
            world_points_3D = ImageRegistration.triangulate_landmarks(
                self.camera_extrinsic_poses[i],
                self.camera_extrinsic_poses[i+1],
                points_1,
                points_2,
                self.camera_intrinsics)

            self.world_points_3D.append(world_points_3D)

    ################################## Helper methods for specific tasks ########################################
    @staticmethod
    def get_essential_matrix(F, K):
        return K.T@F@K

    @staticmethod
    def get_camera_matrix(pose, K):
        P = K@pose[0:3, 0:4]
        return P

    @staticmethod
    def get_transformation_matrix(camera_projection_matrix):
        return np.linalg.inv(camera_projection_matrix)

    @staticmethod
    def triangulate_landmarks(projection_1, projection_2, points_1, points_2, K):
        P1 = ImageRegistration.get_camera_matrix(projection_1, K)
        P2 = ImageRegistration.get_camera_matrix(projection_2, K)
        world_points_3D = cv2.triangulatePoints(P1, P2, points_1.reshape(-1, 1, 2), points_2.reshape(-1, 1, 2))

        # Convert to Euclidean coordinates by dividing by the 4th coordinate.
        world_points_3D[:3, :] = world_points_3D[:3, :]/world_points_3D[3, :]

        # Compute the transform from the camera frame to world frame.
        T_camera_to_world = np.linalg.inv(projection_1)

        # TODO (KSorte): Review this transformation of landmarks into the world frame.
        # Convert the homogeneous landmark coordinates from the first camera frame to the world frame.
        world_points_3D = T_camera_to_world@world_points_3D
        return world_points_3D

    ############################################# Plotting methods ####################################################
    def drawlines(self, first_index, second_index, lines1, lines2, pts1, pts2):
        """
        Draw epipolar line on image given by first_index.

        Args:
            first_index (int): Index of the first image in self.images.
            second_index (int): Index of the second image in self.images.
            lines1 (numpy.ndarray): Epipolar lines in the first image.
            lines2 (numpy.ndarray): Epipolar lines in the second image.
            pts1 (numpy.ndarray): Keypoints in the first image.
            pts2 (numpy.ndarray): Corresponding keypoints in the second image.

        Returns:
            tuple: Two images with epipolar lines and points drawn on them.
        """
        img1 = self.images[first_index].copy()
        img2 = self.images[second_index].copy()

        # Get image dimensions
        _, cols = img1.shape[:2]

        for line1, line2, pt_in_img1, pt_in_img2 in zip(lines1, lines2, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # -------------------------- Line on Image 1 -------------------------
            # Calculate the endpoints of the epipolar line
            # Get the y intercept.
            x0, y0 = map(int, [0, -line1[2] / line1[1]])
            # Get the y coordinate when the line1 hits the last column (last x.)
            x1, y1 = map(int, [cols, -(line1[2] + line1[0] * cols) /line1[1]])

            # Draw the epipolar line on img1
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)

            pt_in_img1 = tuple(map(int, pt_in_img1.ravel()))

            # Draw circles for the keypoints
            img1 = cv2.circle(img1, pt_in_img1, 10, color, -1)

            # ---------------------------- Line on Image 2 ------------------------
            # Calculate the endpoints of the epipolar line
            # Get the y intercept.
            x0, y0 = map(int, [0, -line2[2] / line2[1]])
            # Get the y coordinate when the line2 hits the last column (last x.)
            x1, y1 = map(int, [cols, -(line2[2] + line2[0] * cols) /line2[1]])

            # Draw the epipolar line on img2
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 3)

            pt_in_img2 = tuple(map(int, pt_in_img2.ravel()))

            # Draw circles for the keypoints
            img2 = cv2.circle(img2, pt_in_img2, 10, color, -1)

        return img1, img2

    def compute_and_draw_epipolar_lines(self, first_index, second_index, inlier_matches, F):
        """
        Visualizes epipolar lines between two images.

        Args:
            first_index (int): Index of the first image in self.images.
            second_index (int): Index of the second image in self.images.
        """

        # Get coordinates of the matched keypoints from both images
        pts1 = np.float32([self.keypoints[first_index][m[0].queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.keypoints[second_index][m[0].trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)

        # Compute epilines for points in second image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)

        # Compute epilines for points in first image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)

        # Draw epilines and points
        img1, img2 = self.drawlines(first_index, second_index, lines1, lines2, pts1, pts2)

        # Display results
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar lines on image {first_index + 1}')
        plt.subplot(122), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar lines on image {second_index + 1}')
        plt.show()

    def plot_camera_poses(self, axis_length=0.1):
        """
        Plots all camera poses in 3D.

        Each pose is represented by a coordinate frame with colored axes indicating orientation.

        Parameters:
            axis_length (float): Length of the axis lines for each camera frame.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set plot limits (adjust as needed for your data)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Plot each camera pose
        for i, pose in enumerate(self.camera_extrinsic_poses):

            # Extract rotation and translation from the 4x4 pose matrix
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Define the origin of the camera in world coordinates
            origin = t

            # Define the axes of the camera frame in world coordinates
            x_axis = origin + axis_length * R[:, 0]  # X-axis in red
            y_axis = origin + axis_length * R[:, 1]  # Y-axis in green
            z_axis = origin + axis_length * R[:, 2]  # Z-axis in blue

            # Plot the camera origin
            ax.scatter(*origin, color="black", s=20)

            # # Plot the axes of the camera
            ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
            ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
            ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

            # Optional: Label the camera frame
            ax.text(origin[0], origin[1], origin[2], f"Camera {i+1}", color="black")

        # Show the plot
        plt.show()

    @staticmethod
    def plot_points_3d(world_points_3D):
        """
        Simple 3D scatter plot of triangulated points.

        Parameters:
        world_points_3D: list of numpy.ndarray
            List of 4xN arrays of homogeneous 3D points from triangulation
        """
        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points from each image pair
        for points_4d in world_points_3D:
            points_3d = points_4d[:3]

            # Plot the points
            ax.scatter(points_3d[0], points_3d[1], points_3d[2],
                    c='blue', marker='.', s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
