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
                 display_feature_matching = False,
                 sift_nOctaveLayers = 3,
                 sift_contrastThreshold = 0.04,
                 sift_edgeThreshold = 10,
                 sift_sigma = 1.6,
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


        self.display_feature_matching = display_feature_matching

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