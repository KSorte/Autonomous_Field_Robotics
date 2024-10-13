import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import re

class ImageMosiacking:
    """
    A class to perform perform image mosiacking operations on the Skerski dataset using GTSAM.
    """

    def __init__(self, image_directories,
                 display_feature_matching = False,
                 use_blending = False,
                 mosaic_name = "mosiac",
                 link_proposal_distance_factor = 1.5,
                 max_reprojection_error = 2.0,
                 min_inliers = 25):
        """
        Initializes the ImageMosaicking class.

        Parameters:
        -----------
        image_directories : list
            Directories containing images for mosaicking.
        display_feature_matching : bool, optional
            If True, displays feature matching (default: False).
        use_blending : bool, optional
            Enables blending during mosaicking (default: False).
        mosaic_name : str, optional
            Name of the output mosaic (default: "mosaic").
        link_proposal_distance_factor : float, optional
            Factor to multiply to maximum temporal distance between links for proposing new links (default: 1.5).
        max_reprojection_error : float, optional
            Maximum reprojection error for validating overlaps (default: 2.0).
        min_inliers : int, optional
            Minimum number of inliers for link validation (default: 25).

        Attributes:
        -----------
        - self.sift : cv2.SIFT
            SIFT detector initialized with specific parameters.
        - self.complete_graph : dict
            Stores links, inlier matches, and homographies for images.
        """
        self.images = []

        # Image directory
        self.image_directories = image_directories

        # TODO(KSorte): Either make sift parameters as constructor arguments or write a
        # set sift params function that sets it. Do not edit source code to tune SIFT.
        self.sift = cv2.SIFT.create(nOctaveLayers = 5,
                                    contrastThreshold=0.02,
                                    edgeThreshold = 8.5,
                                    sigma = 0.9)


        self.display_feature_matching = display_feature_matching

        self.mosaic_name = mosaic_name

        # Dictionary that maps the image link with the corresponding inlier matches and homography
        self.complete_graph = {}

        # Maximum reprojection error for non temporal overlaps.
        self.MAX_REPROJECTION_ERROR = max_reprojection_error

        # Min Inliers for link validation.
        self.MIN_INLIERS = min_inliers

        # Distance factor for proposing new links.
        self.LINK_PROPOSAL_DISTANCE_FACTOR = link_proposal_distance_factor

        # Whether to use blending
        self.use_blending = use_blending

    def get_images_for_mosaicking(self):
        """
        Loads all images from multiple subdirectories for mosaicking and displays them.
        Images are read in numerical order based on their filenames.
        """

        # Function to extract numbers from filenames for numerical sorting
        def extract_number(filename):
            match = re.search(r'\d+', filename)  # Find the first number in the filename
            return int(match.group()) if match else float('inf')  # Return inf if no number is found

        self.images = []  # Initialize images list

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
            cols = 3  # You can adjust the number of columns based on preference
            rows = math.ceil(num_images / cols)

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop through the images and plot each in the corresponding grid cell
        for i, image in enumerate(self.images):
            # Convert BGR to RGB for display in Matplotlib
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[i].axis('off')  # Turn off axis for a cleaner look

        # Turn off any unused axes (if the grid is larger than the number of images)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()

    def get_features(self):
        """
        Detects SIFT keypoints and descriptors for each image in `self.images`, stores them in
        `self.keypoints` and `self.descriptors`, and visualizes the keypoints on the grayscale images.

        Keypoints and descriptors are extracted from each image after conversion to grayscale for
        efficiency. The detected keypoints are drawn on the images and displayed using Matplotlib.

        Attributes:
            self.keypoints (list): Keypoints detected in each image.
            self.descriptors (list): Corresponding descriptors for the keypoints.

        """
        self.keypoints = []
        self.descriptors = []
        i = 0
        for image in self.images:

            # Convert image to grayscale to reduce the computational load on SIFT by reducing
            # intensity to a single channel.
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get keypoints and descriptors.
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)

            # Draw and plot the keypoints
            image_with_keypoints = cv2.drawKeypoints(
                gray_image, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if self.display_feature_matching:
                plt.figure(figsize=(10, 6))
                plt.imshow(image_with_keypoints, cmap='gray')
                plt.title(f'Image {i+1} with Keypoints')
                plt.axis('off')
                plt.show()
            i += 1

    def match_features(self, index1, index2):
        """
        Detects SIFT keypoints and descriptors for each image in `self.images` and stores them
        in `self.keypoints` and `self.descriptors`. Each image is converted to grayscale to
        optimize feature detection. The method also visualizes and plots the keypoints on the
        images using Matplotlib.

        Attributes:
            self.keypoints (list): List of keypoints for each image.
            self.descriptors (list): List of descriptors for each image.

        Procedure:
            1. Convert each image to grayscale.
            2. Detect keypoints and descriptors using SIFT.
            3. Append results to `self.keypoints` and `self.descriptors`.
            4. Visualize keypoints using `cv2.drawKeypoints()` and Matplotlib.

        Example:
            >>> obj.get_features()
        """
        brute_force_matcher = cv2.BFMatcher()

        # Get matches between the two images, alongwith a second best match for each batch.
        matches = brute_force_matcher.knnMatch(self.descriptors[index1], self.descriptors[index2], k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good_matches.append([m])

        return good_matches

    def get_matches_for_temporal_sequence_and_homographies(self):
        """
        Computes feature matches and homographies between consecutive images in a sequence.

        Matches keypoints between consecutive images, visualizes the matches, and computes the
        homography matrix for each image pair using RANSAC. The homographies are adjusted
        relative to the first image.

        Attributes:
            self.image_feature_matches (list): Stores lists of good matches between consecutive images.
            self.homographies (list): Stores the homographies computed between consecutive image pairs.
        """
        self.homographies = []
        # Assign identity matrix
        self.homographies.append(np.eye(3))
        self.min_inlier_matches = float('inf')

        for i in range(len(self.images) - 1):
            # Get matches for i and i + 1 th image
            good_matches = self.match_features(i+1, i)
            H, inlier_matches = self.compute_homography(i+1, i, good_matches=good_matches)

            # Store homography and matches.
            self.homographies.append(H)

            # Add link with i+1 as source and i as the destination.
            # NOTE: Store the NON ADJUSTED homographies.
            self.complete_graph[(i+1, i)] = [H, inlier_matches]

            # Reprojection error
            self.complete_graph[(i+1, i)].append(self.compute_reprojection_error(i+1, i, H, inlier_matches))

        self.adjusted_homographies = []
        # Assign identity transform to the first image.
        self.adjusted_homographies.append(np.eye(3))
        # Adjust homographies with respect to the first image.
        for i in range(1, len(self.homographies)):
            # Adjust homographies
            mosiacing_H = self.adjusted_homographies[i-1] @ self.homographies[i]
            self.adjusted_homographies.append(mosiacing_H)

    def compute_homography(self, src_index, dest_index, good_matches):
        """
        Computes the homography matrix between two images using matched keypoints.

        The function takes matched keypoints between two images, normalizes the coordinates to
        be within the range [-1, 1] for more robust homography estimation, and applies RANSAC
        to compute the homography matrix. The normalization transformation is then inverted
        and combined with the homography for the final result.

        Args:
            src_index (int): Index of the source image in `self.images`.
            dest_index (int): Index of the destination image in `self.images`.
            good_matches (list of cv2.DMatch): List of good matches between the two images.

        Returns:
            numpy.ndarray: The computed homography matrix (3x3).

        Procedure:
            1. Extracts the matched keypoints' coordinates from both source and destination images.
            2. Normalizes the coordinates to the range [-1, 1].
            3. Computes the homography using RANSAC to filter out outliers.
            4. Applies normalization transformation to the homography.
        """

        # Get coordinates of the matched keypoints from both images.
        src_pts = np.float32([self.keypoints[src_index][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dest_pts = np.float32([self.keypoints[dest_index][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Normalize feature coordinates.
        # height, width, _ = self.images[src_index].shape
        # dest_pts[:, 0, 0] = np.divide(dest_pts[:, 0, 0], height / 2) - 1
        # dest_pts[:, 0, 1] = np.divide(dest_pts[:, 0, 1], width / 2) - 1

        # src_pts[:, 0, 0] = np.divide(src_pts[:, 0, 0], height / 2) - 1
        # src_pts[:, 0, 1] = np.divide(src_pts[:, 0, 1], width / 2) - 1

        # Compute homography using RANSAC.
        affine_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dest_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        H = np.vstack([affine_matrix, [0, 0, 1]])


        # Separate good matches into inliers and outliers based on the mask.
        matches_mask = mask.ravel().tolist()
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
        outlier_matches = [good_matches[i] for i in range(len(good_matches)) if not matches_mask[i]]

        if self.display_feature_matching:
            # Draw matches before RANSAC.
            before_ransac_img = cv2.drawMatchesKnn(
                self.images[src_index], self.keypoints[src_index],
                self.images[dest_index], self.keypoints[dest_index],
                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Draw matches after RANSAC (inliers only).
            after_ransac_img = cv2.drawMatchesKnn(
                self.images[src_index], self.keypoints[src_index],
                self.images[dest_index], self.keypoints[dest_index],
                inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Plot both images side by side.
            plt.figure(figsize=(15, 10))

            # Display the matches before RANSAC.
            plt.subplot(1, 2, 1)
            plt.imshow(before_ransac_img)
            plt.title(f'Matches Before RANSAC (Images {src_index} and {dest_index})')
            plt.axis('off')

            # Display the matches after RANSAC.
            plt.subplot(1, 2, 2)
            plt.imshow(after_ransac_img)
            plt.title(f'Matches After RANSAC (Images {src_index} and {dest_index})')
            plt.axis('off')

            plt.show()

        # Normalization matrices.
        # normalization_matrix = np.array([[2 / height, 0, -1],
        #                                 [0, 2 / width, -1],
        #                                 [0, 0, 1]])
        # normalization_matrix_inv = np.linalg.inv(normalization_matrix)

        # Return the computed homography.
        # return normalization_matrix_inv @ H @ normalization_matrix, inlier_matches
        return H, inlier_matches

    @staticmethod
    def get_2D_pose_from_homography(H):
        """
        Extract the 2D rotation matrix and translation vector from a 3x3 homography matrix.
        The underlying assumption that the robot motion and scene (shipwreck) are completely planar.

        Parameters:
        H (numpy.ndarray): 3x3 homography matrix

        Returns:
        R_2D (numpy.ndarray): 2x2 rotation matrix
        T_2D (numpy.ndarray): 2D translation vector
        """
        # Normalize the homography matrix by dividing by H[2, 2] (to remove any scaling)
        H = H / H[2, 2]

        # Extract the 2x2 rotation matrix from the top-left corner of the homography matrix
        R_2D = H[:2, :2]

        # Get yaw from rotation matrix
        yaw = np.arctan2(R_2D[1, 0], R_2D[0, 0])

        # Extract the 2D translation vector from the third column of the homography matrix
        T_2D = H[:2, 2]

        return T_2D, yaw

    def compute_reprojection_error(self, i, j, H, inlier_matches):
        """
        Computes the mean reprojection error between matched feature points.

        Parameters:
        -----------
        i : int
            Index of the source image.
        j : int
            Index of the destination image.
        H : np.ndarray
            Homography matrix between the two images.
        inlier_matches : list
            List of inlier feature matches between the images.

        Returns:
        --------
        float
            The mean Euclidean distance between the actual and projected points.
        """
        # Get feature points from the inlier matches.
        src_pts = np.float32([self.keypoints[i][m[0].queryIdx].pt for m in inlier_matches])
        dest_pts = np.float32([self.keypoints[j][m[0].trainIdx].pt for m in inlier_matches])

        # Convert points to homogeneous coordinates
        src_pts_homog = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))

        # Project the source points using the homography matrix
        projected_pts_homog = np.dot(H, src_pts_homog.T).T

        # Convert back to Cartesian coordinates by dividing by the homogeneous coordinate
        projected_pts = projected_pts_homog[:, :2] / projected_pts_homog[:, 2:]

        # Compute the Euclidean distance between the actual and projected points
        errors = np.linalg.norm(dest_pts - projected_pts, axis=1)

        # Return the mean reprojection error
        return np.mean(errors)

    @staticmethod
    def transform_image_corners(image, H):
        """
        Transforms the corner points of an image using the given homography matrix.

        Args:
            image (ndarray): The input image whose corners are to be transformed.
            H (ndarray): The homography matrix.

        Returns:
            tuple: Transformed corner points and (xmin, ymin, xmax, ymax) coordinates.
        """

        # Get image corner points coordinates.
        img_corner_points = np.array(
            [[0, 0],
             [image.shape[1], 0],
             [image.shape[1],image.shape[0]],
             [0, image.shape[0]]], dtype=np.float32)

        # Transform and reshape.
        transformed_points = cv2.perspectiveTransform(img_corner_points.reshape((1, -1, 2)), H)
        transformed_points = transformed_points.reshape(-1, 2)

        xcoords = transformed_points[:, 0]
        ycoords = transformed_points[:, 1]

        # Find xmin, xmax, ymin, ymax
        xmin = np.min(xcoords)
        xmax = np.max(xcoords)
        ymin = np.min(ycoords)
        ymax = np.max(ycoords)
        return transformed_points, xmin, ymin, xmax, ymax

    # Get size of the canvas on which panorama displayed
    def get_panorama_canvas(self):
        """
        Computes the size of the canvas for the image mosiac based on the transformed corner points of all images.

        Notes:
            - Stores the transformed corner points for all images.
            - Calculates the minimum and maximum coordinates to determine canvas size.
        """
        # Get image corner points after transforming images on central plane.
        self.transformed_img_corners = []

        self.xmin_canvas = float('inf')
        self.ymin_canvas = float('inf')
        self.xmax_canvas = float('-inf')
        self.ymax_canvas = float('-inf')

        for i in range(len(self.images)):
            img_corners, xmin, ymin, xmax, ymax = ImageMosiacking.transform_image_corners(
                 self.images[i], self.adjusted_homographies[i])

            self.xmin_canvas = min(xmin, self.xmin_canvas)
            self.ymin_canvas = min(ymin, self.ymin_canvas)
            self.xmax_canvas = max(xmax, self.xmax_canvas)
            self.ymax_canvas = max(ymax, self.ymax_canvas)
            self.transformed_img_corners.append(img_corners)

        self.canvas_x_size = math.ceil(self.xmax_canvas - self.xmin_canvas)
        self.canvas_y_size = math.ceil(self.ymax_canvas - self.ymin_canvas)

    def project_images_on_canvas(self, mosiac_name = "mosiac"):
        """
        Maps the images using their homographies into the image plane of the central image.
        Now includes dynamic blending for overlapping regions.
        """
        # Initialize empty canvas
        self.canvas = np.zeros((self.canvas_y_size, self.canvas_x_size, 3), dtype=np.uint8)
        self.canvas[:] = (0, 0, 0)

        # Translation matrix
        T = np.array([[1, 0, abs(self.xmin_canvas)],
                    [0, 1, abs(self.ymin_canvas)],
                    [0, 0, 1]])

        # Create a mask to track where we have already placed images on the canvas
        canvas_mask = np.zeros((self.canvas_y_size, self.canvas_x_size), dtype=np.uint8)

        for i in range(len(self.images)):
            H = self.adjusted_homographies[i]
            img = self.images[i]

            # Warp the image onto the canvas
            warped_img = cv2.warpPerspective(np.copy(img), T @ H, (self.canvas_x_size, self.canvas_y_size))

            # Get the corners of the warped image
            corner_pts_warped_img, _, _, _, _ = ImageMosiacking.transform_image_corners(self.images[i], T @ H)
            corner_pts_warped_img = corner_pts_warped_img.reshape(-1, 1, 2)

            # Create a mask for the warped image
            warped_img_mask = np.zeros((self.canvas_y_size, self.canvas_x_size), dtype=np.uint8)
            cv2.fillPoly(warped_img_mask, [corner_pts_warped_img.astype(int)], 255)

            # Find the overlapping area (intersection of current image with canvas)
            overlap_mask = cv2.bitwise_and(warped_img_mask, canvas_mask)

            # For the overlapping region, apply dynamic blending using distance transforms
            if np.sum(overlap_mask) > 0:
                overlap_indices = np.where(overlap_mask > 0)

                if self.use_blending:
                    # Distance transform to calculate the proximity to the edge of each mask
                    dist_canvas = cv2.distanceTransform(canvas_mask, cv2.DIST_L2, 5)
                    dist_warped = cv2.distanceTransform(warped_img_mask, cv2.DIST_L2, 5)

                    # Normalize distances to avoid division by zero
                    dist_sum = dist_canvas[overlap_indices] + dist_warped[overlap_indices] + 1e-6

                    # Calculate alpha blending dynamically based on distance
                    alpha = dist_warped[overlap_indices] / dist_sum
                else:
                    # alpha is 1. No blending.
                    alpha = np.ones(len(overlap_indices[0]))

                # Blend the overlapping pixels dynamically
                self.canvas[overlap_indices] = (self.canvas[overlap_indices].astype(np.float32) * (1 - alpha[:, np.newaxis]) +
                                                warped_img[overlap_indices].astype(np.float32) * alpha[:, np.newaxis]).astype(np.uint8)

            # Update the non-overlapping parts of the canvas
            non_overlap_mask = cv2.bitwise_and(warped_img_mask, cv2.bitwise_not(canvas_mask))
            self.canvas = cv2.bitwise_or(self.canvas, cv2.bitwise_and(warped_img, warped_img, mask=non_overlap_mask))

            # Update the canvas mask to include the new image's region
            canvas_mask = cv2.bitwise_or(canvas_mask, warped_img_mask)

        # Plot canvas with blended images
        plt.imshow(self.canvas)
        plt.imsave(mosiac_name + '.JPG', self.canvas)
        plt.show()


    def get_temporal_trajectory(self):
        """
        Computes and plots the temporal trajectory of image centers using homographies.

        Updates the image center trajectory by iterating through homographies and
        extracting 2D translation and yaw between consecutive images.

        Returns:
        --------
        None
        """
        self.image_center_trajectory = np.zeros((len(self.images), 3))
        for i, H in enumerate(self.homographies):
            if i == 0:
                continue
            # Get the rotation and translation from Homography from i to i+1
            T_2D, yaw = ImageMosiacking.get_2D_pose_from_homography(H)
            # X Coordinate
            self.image_center_trajectory[i, 0] = self.image_center_trajectory[i-1, 0] - T_2D[0]
            # Y Coordinate
            self.image_center_trajectory[i, 1] = self.image_center_trajectory[i-1, 1] - T_2D[1]
            # Heading
            self.image_center_trajectory[i, 2] = self.image_center_trajectory[i-1, 2] - yaw

        # Plot the trajectory
        plt.figure()
        plt.plot(self.image_center_trajectory[:, 0], self.image_center_trajectory[:, 1], 'bo-', label='Trajectory')

        # Annotate each point with the image number
        for i, (x, y, yaw) in enumerate(self.image_center_trajectory):
            plt.text(x, y, f'{i}', fontsize=12, color='red', ha='right')  # Display image number

        plt.xlabel('X (in reference frame)')
        plt.ylabel('Y (in reference frame)')
        plt.title('Temporal Trajectory of Image Centers in Reference Frame')
        plt.legend()
        plt.grid(True)
        plt.show()

    def update_homographies(self, pose_trajectory):
        """
        Converts the numpy poses into 3x3 homography matrices using the yaw for rotation and
        x, y for translation. Overwrites the homographies in self.adjusted_homographies.

        Parameters:
        pose_trajectory (np.ndarray): Array of poses where each row contains [x, y, yaw].
        """
        # Iterate over each pose and update the homographies
        for i, (x, y, yaw) in enumerate(pose_trajectory):
            # 3x3 homography matrix
            homography = np.eye(3)

            # Rotation part using yaw
            homography[0, 0] = np.cos(yaw)
            homography[0, 1] = np.sin(yaw)
            homography[1, 0] = -np.sin(yaw)
            homography[1, 1] = np.cos(yaw)

            # Translation
            homography[0, 2] = -x
            homography[1, 2] = -y

            # Update the ith homography
            self.adjusted_homographies[i] = homography

    def plot_all_links(self):
        """
        Plots the temporal trajectory of image centers and valid non-temporal links.

        Displays the trajectory with annotated image numbers and overlays
        validated links between non-temporal images.

        Returns:
        --------
        None
        """
        # Plot the trajectory
        plt.figure()

        # Plot trajectory (with label added for the legend)
        plt.plot(self.image_center_trajectory[:, 0], self.image_center_trajectory[:, 1], 'bo-', label='Temporal Trajectory')

        # Annotate each point with the image number
        for i, (x, y, yaw) in enumerate(self.image_center_trajectory):
            plt.text(x, y, f'{i}', fontsize=12, color='red', ha='right')  # Display image number

        # Choose a fixed color for all the proposed links
        link_color = 'green'

        # Plot additional lines for the proposed links
        for (i, j) in self.non_temporal_links:
            x_values = [self.image_center_trajectory[i, 0], self.image_center_trajectory[j, 0]]
            y_values = [self.image_center_trajectory[i, 1], self.image_center_trajectory[j, 1]]

            # Plot the link with a fixed color
            plt.plot(x_values, y_values, color=link_color, linestyle='--')

        # Adding a label for the proposed links for the legend (just once)
        plt.plot([], [], color=link_color, linestyle='--', label='Valid Non Temporal Links')

        # Finalizing the plot
        plt.xlabel('X (in reference frame)')
        plt.ylabel('Y (in reference frame)')
        plt.title('All Overlaps considered for Optimization.')

        # Add legend for both trajectory and proposed links
        plt.legend()

        plt.grid(True)
        plt.show()


    # TODO(KSorte): Write a establish non temporal links function
    # that calls propose and validate non temporal links.
    def propose_non_temporal_links(self):
        """
        Proposes non-temporal links between images based on spatial distance in the temporal trajectory.

        Identifies and stores links between image pairs whose distance is less than a
        threshold based on the maximum temporal distance, excluding adjacent temporal links.

        Returns:
        --------
        None
        """
        # Compute max temporal distance
        max_distance = float('-inf')
        for i in range(len(self.image_center_trajectory) - 1):
            point1 = self.image_center_trajectory[i, :]
            point2 = self.image_center_trajectory[i+1, :]
            distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            max_distance = max(distance, max_distance)

        # Storing proposed links as a set of tuples.
        self.proposed_links = set()
        # Propose all links that are at a lesser distance than the maximum distance.
        for i in range(len(self.images)):
            for j in range(len(self.images)):
                if j == i-1 or j == i+1 or i == j:
                    # Temporal links.
                    continue

                if (j, i) in self.proposed_links:
                    # Link exists.
                    continue

                point_i = self.image_center_trajectory[i, :]
                point_j = self.image_center_trajectory[j, :]
                distance = math.sqrt((point_j[0] - point_i[0])**2 + (point_j[1] - point_i[1])**2)

                if distance < self.LINK_PROPOSAL_DISTANCE_FACTOR*max_distance:
                    # Propose a new link. i -> j
                    self.proposed_links.add((i, j))

    def validate_proposed_links(self):
        """
        Validates proposed non-temporal links using a max reprojection error and minimum number of inlier matches.

        Removes links with high reprojection error or insufficient inlier matches,
        and stores valid links along with their homography, inliers, and error in the complete graph.

        Returns:
        --------
        None
        """
        # Iterate over a copy of the proposed_links set to avoid modifying it while iterating.
        self.non_temporal_links = self.proposed_links.copy()
        for link in self.proposed_links:
            i, j = link
            good_matches = self.match_features(i, j)

            # Compute homography with i as the source and j as the destination
            H, inlier_matches = self.compute_homography(i, j, good_matches=good_matches)

            # Get reprojection error for this homography.
            reprojection_error = self.compute_reprojection_error(i, j, H, inlier_matches)

            # Remove the link if the number of inlier matches is less than the threshold.
            if reprojection_error > self.MAX_REPROJECTION_ERROR or len(inlier_matches) < self.MIN_INLIERS:
                # Bad homography.
                self.non_temporal_links.remove((i, j))
                continue

            self.complete_graph[(i, j)] = [H, inlier_matches, reprojection_error]

    def align_images_in_temporal_sequence(self, mosiac_name = "mosiac"):
        """
        Gets image features, matches and homographies.
        Renders the panorama by warping each image using its homography and placing it on the canvas.
        An overarching function that calls the compute homographies, computing canvas and image projection operations.
        """
        self.get_features()
        self.get_matches_for_temporal_sequence_and_homographies()
        self.get_panorama_canvas()
        self.project_images_on_canvas(mosiac_name=mosiac_name)
