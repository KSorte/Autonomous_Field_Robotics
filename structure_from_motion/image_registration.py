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


        self.visualize_features = visualize_features
        self.visualize_epipolar_lines = visualize_epipolar_lines
        self.ransac_reproj_thres = ransac_reproj_thres

    def get_images_for_registration(self):
        """
        Loads all images from multiple subdirectories for mosaicking and displays them.
        Images are read in numerical order based on their filenames.
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

        # Set up the grid layout for plotting
        num_images = len(self.images)
        # Number of columns in the grid.
        num_cols = 4
        # Calculate number of rows needed
        num_rows = (num_images + num_cols - 1) // num_cols

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

            # Draw the keypoints on the grayscale image
            image_with_keypoints = cv2.drawKeypoints(
                gray_image, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if self.visualize_features:
                # Plot the image with keypoints in the grid
                axes[i].imshow(image_with_keypoints, cmap='gray')
                axes[i].set_title(f'Image {i+1} with Keypoints')
                axes[i].axis('off')

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
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good_matches.append([m])

        return good_matches


    def compute_fundamental_matrix(self, first_index, second_index, good_matches):
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
        first_pts = first_pts[mask.ravel()==1]
        second_pts = second_pts[mask.ravel()==1]

        # Separate good matches into inliers and outliers based on the mask.
        matches_mask = mask.ravel().tolist()
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]

        return F, inlier_matches

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

    def get_matches_and_fundamental_matrices(self):
        """
        Computes feature matches and fundamental matrices between consecutive images in a sequence.

        Matches keypoints between consecutive images, visualizes the matches, and computes the
        fundamental matrix for each image pair using RANSAC.

        Attributes:
            self.image_feature_matches (list): Stores lists of good matches between consecutive images.
            self.fundamental_matrices (list): Stores the homographies computed between consecutive image pairs.
        """
        self.fundamental_matrices = []

        for i in range(len(self.images) - 1):
            # Get matches for i and i + 1 th image
            good_matches = self.match_features(i, i+1)
            # Get Fundamental Matrix and inlier matches.
            F, inlier_matches = self.compute_fundamental_matrix(i, i+1, good_matches=good_matches)

            # Store homography and matches.
            self.fundamental_matrices.append(F)

            if self.visualize_epipolar_lines:
                # Compute and Visualize epipolar lines
                self.compute_and_draw_epipolar_lines(i, i+1, inlier_matches, F)