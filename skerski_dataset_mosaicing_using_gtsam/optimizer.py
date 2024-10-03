import gtsam
import matplotlib.pyplot as plt
import numpy as np
import image_mosiacing as im

class GTSAMOptimizer:
    def __init__(self,image_mosiacking_obj):
        self.img_mos = image_mosiacking_obj
        self.poses = []
        self.initial_estimate = gtsam.Values()

    # Add prior for the first pose (anchoring the graph)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

    def add_initial_estimates_from_trajectory(self):
        """
        Add initial estimates for the poses using the image center trajectory.
        Assumes that the yaw (theta) is initially 0 for all poses.
        """

        # Loop through the image center trajectory
        rows, cols = self.img_mos.image_center_trajectory.shape
        for i in range(rows):
            # Extract x and y coordinates and heading from the trajectory
            x, y, yaw = self.img_mos.image_center_trajectory[i, 0], \
                        self.img_mos.image_center_trajectory[i, 1], \
                        self.img_mos.image_center_trajectory[i, 2]

            # Create a Pose2 object with (x, y) and a zero initial orientation (yaw)
            initial_pose = gtsam.Pose2(x, y, yaw)

            # Add the pose to the initial estimate
            self.initial_estimate.insert(i, initial_pose)

    def build_graph(self):
        self.graph = gtsam.NonlinearFactorGraph()

        start_x = self.img_mos.image_center_trajectory[0, 0]
        start_y = self.img_mos.image_center_trajectory[0, 1]

        # Prior Factor
        self.graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(start_x, start_y, 0), self.prior_noise))

        self.add_initial_estimates_from_trajectory()

        for link in self.img_mos.complete_graph.keys():
            # Get link coordinates.
            i, j = link
            # Reprojection error
            error = self.img_mos.complete_graph[link][2]

            numMatches = len(self.img_mos.complete_graph[link][1])
            if numMatches > 200:
                err = [0.2, 0.2, 0.1] # Minimum Error
            elif numMatches > 100 and numMatches <= 200:
                err = [0.5, 0.5, 0.2]
            elif numMatches > 50 and numMatches <= 100:
                err = [0.8, 0.8, 0.4]
            else:
                err = [1.0, 1.0, 0.5] # Maximum Error
            # Noise model based off the reprojection error.
            err = [error*2, error*2, error*2*3.141592653589793238/180]
            # err = [10.0, 10.0, 10*3.14/180]
            # if abs(i - j) > 1:
            #     # Smaller covariance for non temporal
            #     err = [2.0, 2.0, 2*3.14/180]

            noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array(err))

            i, j = link

            H = self.img_mos.complete_graph[link][0]
            # T_2D, yaw = self.get_2D_pose_from_homography(H)
            T_2D, yaw = im.ImageMosiacking.get_2D_pose_from_homography(H)

            # Create the relative pose (Pose2 object) from translation and yaw
            relative_pose = gtsam.Pose2(T_2D[0], T_2D[1], yaw)


            # Add BetweenFactorPose2 for the link (i, j)
            self.graph.add(gtsam.BetweenFactorPose2(i, j, relative_pose, noise_model))

    def optimize_graph(self):
        """
        Optimizes the factor graph using GTSAM's Levenberg-Marquardt optimizer.

        Parameters:
        graph (gtsam.NonlinearFactorGraph): The built GTSAM factor graph.
        initial_estimate (gtsam.Values): The initial estimate for poses.

        Returns:
        gtsam.Values: The optimized values for the poses.
        """
        params = gtsam.LevenbergMarquardtParams()
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        # Optimize the graph
        self.result = self.optimizer.optimize()

    def plot_optimized_trajectory(self):
        """
        Plots the optimized trajectory of poses.
        """
        # Create lists to store x, y, and yaw values for the poses
        x_vals = []
        y_vals = []
        yaw_vals = []

        # Iterate through the optimized poses
        for i in range(self.result.size()):
            pose = self.result.atPose2(i)
            x_vals.append(pose.x())
            y_vals.append(pose.y())
            yaw_vals.append(pose.theta())

        # Plot the positions (x, y)
        plt.figure()
        plt.plot(x_vals, y_vals, 'bo-', label='Optimized Trajectory')

        # Plot the orientations (yaw) as arrows
        for idx, (x, y, yaw) in enumerate(zip(x_vals, y_vals, yaw_vals)):
            # Arrow to represent orientation (yaw)
            dx = np.cos(yaw) * 0.5  # Scale for visualization
            dy = np.sin(yaw) * 0.5
            plt.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='red', ec='red')

            # Annotate with the pose number
            plt.text(x, y, str(idx), fontsize=12, color='blue')  # Add pose number next to each pose

        # Add labels and title
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Optimized Poses and Orientations')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()


    def plot_combined_trajectory_and_poses(self):
        """
        Plots the temporal trajectory of image centers and the optimized poses with orientations
        on the same graph.
        """

        # Create a figure for the combined plot
        plt.figure()

        # --- Plot 1: Temporal Trajectory of Image Centers ---
        plt.plot(self.img_mos.image_center_trajectory[:, 0], self.img_mos.image_center_trajectory[:, 1], 'bo-', label='Image Center Trajectory')

        # Annotate each point with the image number
        for i, (x, y, yaw) in enumerate(self.img_mos.image_center_trajectory):
            plt.text(x, y, f'{i}', fontsize=12, color='red', ha='right')  # Display image number

        # --- Plot 2: Optimized Poses and Orientations ---
        # Create lists to store x, y, and yaw values for the poses
        x_vals = []
        y_vals = []
        yaw_vals = []

        # Iterate through the optimized poses
        for i in range(self.result.size()):
            pose = self.result.atPose2(i)
            x_vals.append(pose.x())
            y_vals.append(pose.y())
            yaw_vals.append(pose.theta())
        plt.plot(x_vals, y_vals, 'go-', label='Optimized Trajectory')  # Use a different color for the optimized trajectory

        # Plot the orientations (yaw) as arrows
        for idx, (x, y, yaw) in enumerate(zip(x_vals, y_vals, yaw_vals)):
            # Arrow to represent orientation (yaw)
            dx = np.cos(yaw) * 0.5  # Scale for visualization
            dy = np.sin(yaw) * 0.5
            plt.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='red', ec='red')  # Arrows to indicate yaw

            # Annotate with the pose number
            plt.text(x, y, str(idx), fontsize=12, color='blue')  # Add pose number next to each pose

        # Add labels, title, and legend
        plt.xlabel('X Position (in reference frame)')
        plt.ylabel('Y Position (in reference frame)')
        plt.title('Combined Plot of Image Centers and Optimized Poses')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
