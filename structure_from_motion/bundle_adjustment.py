import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import image_registration as ir
from gtsam import (Cal3_S2, DoglegOptimizer,
                         GenericProjectionFactorCal3_S2, Marginals,
                         NonlinearFactorGraph, PinholeCameraCal3_S2, Point3,
                         Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values)
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X

class GTSAMBundleAdjustment:
    def __init__(self, image_registration_object: ir.ImageRegistration):
        self.img_reg_obj = image_registration_object

        # Camera intrinsics.
        K = self.img_reg_obj.camera_intrinsics
        self.GTSAM_camera_intrinsics = Cal3_S2(K[0,0], K[1, 1], 0, K[0, 2], K[1, 2])

        self.graph = NonlinearFactorGraph()

        # List to store the count of all the landmarks.
        self.all_landmarks_count = []

    def define_factor_noises(self):
        # Define the camera observation noise model
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        # Prior pose noise
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))

        # Add prior landmark factor.
        self.point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

    def build_graph(self):
        self.initial = gtsam.Values()

        # Prior Pose Factor.
        prior_factor = PriorFactorPose3(X(0), gtsam.Pose3(), self.pose_noise)
        self.graph.push_back(prior_factor)

        # Add Prior Point factor.
        point_factor = gtsam.PriorFactorPoint3(
        L(0), self.img_reg_obj.world_points_3D[0][0:3, 0], self.point_noise)
        self.graph.push_back(point_factor)

        # Start with an identity pose.
        current_pose = np.eye(4)
        for i in range(len(self.img_reg_obj.images)-1):
            # Get i+1th camera extrinsics.
            camera_projection = self.img_reg_obj.camera_extrinsic_poses[i+1]
            # Get i+1th camera pose.
            camera_pose = ir.ImageRegistration.get_transformation_matrix(camera_projection)

            # Get i and i+1th view keypoints.
            src_points = self.img_reg_obj.inlier_points[0][i]
            dst_points = self.img_reg_obj.inlier_points[0][i+1]

            # Get i and i+1th view keypoint indices.
            src_point_index_list = self.img_reg_obj.inlier_indices[0][i]
            dst_point_index_list = self.img_reg_obj.inlier_indices[0][i+1]

            # Create factors using inlier points and world 3d points for i and i+1 th views.
            for object_index, (src_point, dst_point, src_point_index, dst_point_index) in enumerate(zip(src_points,
                                                                                                        dst_points,
                                                                                                        src_point_index_list,
                                                                                                        dst_point_index_list)):

                landmark_point = self.img_reg_obj.world_points_3D[i][0:3, object_index]
                # Update the object index for the src and dst points.
                self.img_reg_obj.object_index_list[i][src_point_index] = object_index
                self.img_reg_obj.object_index_list[i+1][dst_point_index] = object_index

                # Add landmark initial estimate
                self.initial.insert(L(object_index), landmark_point)

                # Add landmark measurement factor for ith view.
                self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                    src_point, self.measurement_noise, X(i), L(object_index), self.GTSAM_camera_intrinsics))

                # Add landmark measurement factor for the i+1th view.
                self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                    dst_point, self.measurement_noise, X(i+1), L(object_index), self.GTSAM_camera_intrinsics))

                self.all_landmarks_count.append(2)

            # Add Initial pose estimates for i and i+1 views.
            self.initial.insert(X(i), gtsam.Pose3(current_pose))
            self.initial.insert(X(i+1), gtsam.Pose3(camera_pose))

        # print(self.graph)

    def optimize_graph(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        self.result = optimizer.optimize()

        # print(self.result)

    def extract_landmarks_to_array(self):
        landmarks = []

        # Iterate over all keys in the result
        for key in self.result.keys():
            symbol = str(gtsam.Symbol(key))
            # Check if the key corresponds to a landmark ('l')
            if 'l' in symbol:
                # Extract the 3D point
                point = self.result.atPoint3(key)
                # Add homogeneous coordinate
                landmarks.append([point[0], point[1], point[2], 1.0])

        # Convert to a 4xN numpy array
        # Transpose to get 4xN shape
        landmarks_array = np.array(landmarks).T
        self.landmarks_list = []
        self.landmarks_list.append(landmarks_array)
