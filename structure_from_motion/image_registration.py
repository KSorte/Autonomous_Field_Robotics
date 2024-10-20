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
                 ransac_reproj_thres = 5.0,
                 min_inliers = 25):
        """
        Initializes the ImageRegistration class with placeholder values.
        """
        self.images = []

        # Image directory
        self.image_directories = image_directories

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


    def compute_fundamental_matrix(self, first_index, second_index, good_matches):
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
        outlier_matches = [good_matches[i] for i in range(len(good_matches)) if not matches_mask[i]]

        return F, inlier_matches

    def drawlines(self, first_index, second_index, lines, pts1, pts2):
        """
        Draw epipolar lines on images.

        Args:
            first_index (int): Index of the first image in self.images.
            second_index (int): Index of the second image in self.images.
            lines (numpy.ndarray): Epipolar lines.
            pts1 (numpy.ndarray): Keypoints in the first image.
            pts2 (numpy.ndarray): Corresponding keypoints in the second image.

        Returns:
            tuple: Two images with epipolar lines and points drawn on them.
        """
        img1 = self.images[first_index].copy()
        img2 = self.images[second_index].copy()

        # Get image dimensions
        r, c = img1.shape[:2]

        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Calculate the endpoints of the epipolar line
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

            # Draw the epipolar line on img1
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)

            # Draw circles for the keypoints
            pt1 = tuple(map(int, pt1.ravel()))
            pt2 = tuple(map(int, pt2.ravel()))
            img1 = cv2.circle(img1, pt1, 10, color, -1)
            img2 = cv2.circle(img2, pt2, 10, color, -1)

        return img1, img2

    def draw_epipolar_lines(self, first_index, second_index):
        """
        Visualizes epipolar lines between two images.

        Args:
            first_index (int): Index of the first image in self.images.
            second_index (int): Index of the second image in self.images.
        """
        # Get matches
        good_matches = self.match_features(first_index, second_index)

        # Compute fundamental matrix
        F, inlier_matches = self.compute_fundamental_matrix(first_index, second_index, good_matches)

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
        img1, img2 = self.drawlines(first_index, second_index, lines1, pts1, pts2)
        img3, img4 = self.drawlines(second_index, first_index, lines2, pts2, pts1)

        # Display results
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar lines on image {first_index}')
        plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar lines on image {second_index}')
        plt.show()

    def get_matches_and_fundamental_matrices(self):
        """
        Computes feature matches and fundamental matrices between consecutive images in a sequence.

        Matches keypoints between consecutive images, visualizes the matches, and computes the
        fundamental matrix for each image pair using RANSAC.

        Attributes:
            self.image_feature_matches (list): Stores lists of good matches between consecutive images.
            self.homographies (list): Stores the homographies computed between consecutive image pairs.
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
                # Visualize epipolar lines
                self.draw_epipolar_lines(i, i+1)