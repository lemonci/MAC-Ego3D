import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from tqdm import tqdm
from vpr_utils.vpr_model import VPRModel
from torch.nn import functional as F
import gtsam
from scipy.spatial import KDTree
from PIL import Image
    
class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.test = slam.test
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared
        
        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()
        model_ckpt_path = './/submodules/salad/dino_salad.ckpt'
        self.model = self.load_model(model_ckpt_path)
        
        self.keyframe_descriptors = []
        self.keyframe_info = []
        self.similarity_threshold = 0.8
        self.similarity_threshold_inter = 0.85
        # self.similarity_threshold = 0.7 #for ETH3D
        self.ifma = slam.ifma
        self.ifnoise = slam.ifnoise
        torch.manual_seed(42)
        self.lc_freq = slam.lc_freq
        # Camera poses
        self.trajmanager = [] #store GT poses for each disjoint video
        self.poses = [] #store the pose-len of each video
        
        self.mapping_keyframe_info = []
        self.mapping_keyframe_metadata = []
        self.max_frame_number = 0
        for item in self.dataset_path:
            self.trajmanager.append(TrajManager(self.camera_parameters[8],item))
            if len(self.trajmanager[-1].gt_poses) > self.max_frame_number:
                self.max_frame_number = len(self.trajmanager[-1].gt_poses)
            self.poses.append([self.trajmanager[-1].gt_poses[0]])#pose mat is 4*4 homogeneous
            self.mapping_keyframe_info.append([])
            self.mapping_keyframe_metadata.append([])
            
        self.prior_noise = gtsam.noiseModel.Diagonal.Variances(np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002]))
        self.closure_noise = gtsam.noiseModel.Diagonal.Variances(np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02]))
        
        self.finalOptResult = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.loop_detected = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.trackable_opacity_th = slam.trackable_opacity_th
        
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        
        self.__dict__.update(state)
    
    def load_model(self, ckpt_path):
        """Load the VPR model from a checkpoint."""
        model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )
        model.load_state_dict(torch.load(ckpt_path))
        model = model.eval().to('cuda')
        # print(f"Loaded model from {ckpt_path} Successfully!")
        return model
    """
    def input_transform(self, image_size=None):
        # Transformation function for input images.
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        transform_list = []
        if image_size:
            transform_list.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
        transform_list.append(T.ToTensor())
        transform_list.append(T.Normalize(mean=MEAN, std=STD))
        return T.Compose(transform_list)

    def get_descriptor(self, img_numpy, image_size=None, device='cuda'):
        # Convert numpy image (BGR) to descriptor using the VPR model.
        #img_pil = Image.fromarray(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        # Convert the NumPy array (BGR) directly to a tensor and avoid the PIL conversion.
        img_tensor = torch.from_numpy(img_numpy).float() / 255.0  # Scale to [0, 1] range
        img_tensor = img_tensor.permute(2, 0, 1).to(device)  # Convert HWC to CHW
        transform = self.input_transform(image_size)
        img_tensor = transform(img_tensor).unsqueeze(0)  # Add batch dimension

        st_salad = time.time()
        with torch.no_grad():
            descriptor = self.model(img_tensor)#.cpu()  # Move the result to CPU
        ed_salad = time.time()
        print("salad time:", ed_salad - st_salad)
        return descriptor
    """
    def input_transform(self, img_tensor, image_size=None):
        """Apply transformation directly on PyTorch tensors."""
        MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to('cuda')
        STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to('cuda')

        # Resize using F.interpolate if image_size is provided
        if image_size:
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Normalize the image tensor
        # print(img_tensor.shape)
        img_tensor = (torch.tensor(img_tensor).to('cuda') - MEAN) / STD
        return img_tensor

    def get_descriptor(self, img_numpy, image_size=None, device='cuda'):
        """Convert numpy image (BGR) to descriptor using the VPR model."""
        
        # Convert BGR NumPy image to CHW format PyTorch tensor (float in range 0 to 1)
        img_tensor = torch.from_numpy(img_numpy).permute(2, 0, 1).float().div(255.0).to(device)
        
        # Apply the transformation directly using input_transform
        img_tensor = self.input_transform(img_tensor, image_size)

        # Ensure img_tensor is 4D (batch_size, channels, height, width)
        st_salad = time.time()
        # print("img_tensor", img_tensor.size())
        # Run the image through the model without computing gradients
        with torch.no_grad():
            descriptor = self.model(img_tensor)  # Make sure the model receives a 4D tensor

        ed_salad = time.time()
        # print("salad time:", ed_salad - st_salad)

        return descriptor



    def calculate_similarity(self, descriptor1, descriptor2, metric='cosine'):
        """Calculate cosine similarity between two descriptors."""
        if metric == 'cosine':
            descriptor1 = torch.nn.functional.normalize(descriptor1, dim=1)
            descriptor2 = torch.nn.functional.normalize(descriptor2, dim=1)
            similarity = torch.mm(descriptor1, descriptor2.t())  # Cosine similarity
            return similarity.item()
        else:
            raise ValueError("Unsupported metric! Use 'cosine'.")
        
    def rawPose2GTSAMPose(self, rawPose):
        x, y, z = rawPose[0:3, 3]
        R = rawPose[0:3, 0:3]
        qw = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        return gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz),gtsam.Point3(x, y, z))
    
    def GTSAMPose2RawPose(self, pose):
        R = pose.rotation().matrix()  # Get the rotation matrix
        t = pose.translation()  # Get the translation vector

        # Construct the homogeneous transformation matrix
        H = np.eye(4)  # Create a 4x4 identity matrix
        H[:3, :3] = R  # Set the top-left 3x3 block to the rotation matrix
        H[:3, 3] = t  # Set the top-right 3x1 block to the translation vector
        return H
    
    def rotation_matrix_difference(self, R1, R2):
        R_relative = np.dot(np.transpose(R1), R2)
        trace_value = np.trace(R_relative)
        angle = np.arccos((trace_value - 1) / 2)
        return np.degrees(angle)

    def run(self):
        self.tracking()


    def tracking(self):
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()
        for agentIndex in range(len(self.final_pose)):
            print(f"Start Agent{agentIndex} Tracking")
        
            self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path[agentIndex]}/images", agentIndex)
            self.num_images = len(self.rgb_images)
            self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
            self.reg.set_max_knn_distance(self.knn_max_distance)
            if_mapping_keyframe = False
            self.iteration_images = 0
            
            self.total_start_time = time.time()
            pbar = tqdm(total=self.num_images)
            for ii in range(self.num_images):
                self.iter_shared[0] = ii
                current_image = self.rgb_images.pop(0)
                depth_image = self.depth_images.pop(0)
                current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                # Make pointcloud
                points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(depth_image, current_image)
                # GICP
                if self.iteration_images == 0:
                    current_pose = self.poses[agentIndex][-1]
                    #Add first graph
                    if self.ifma == 1:
                        self.initial_estimates.insert(agentIndex * self.max_frame_number, self.rawPose2GTSAMPose(current_pose))
                        self.graph.add(gtsam.PriorFactorPose3(agentIndex * self.max_frame_number , self.rawPose2GTSAMPose(current_pose), self.prior_noise))
                    
                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(
                            "cam/current",
                            rr.Transform3D(translation=self.poses[agentIndex][-1][:3,3],
                                        rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[agentIndex][-1][:3,:3])).as_quat()))
                        )
                        rr.log(
                            "cam/current",
                            rr.Pinhole(
                                resolution=[self.W, self.H],
                                image_from_camera=self.cam_intrinsic,
                                camera_xyz=rr.ViewCoordinates.RDF,
                            )
                        )
                        rr.log(
                            "cam/current",
                            rr.Image(current_image)
                        )
                        
                    # print(current_pose, "\n",self.trajmanager[agentIndex].gt_poses[ii],"\n---------------------------")
                    current_pose = np.linalg.inv(current_pose)
                    T = current_pose[:3,3]
                    R = current_pose[:3,:3].transpose()
                    
                    # transform current points
                    points_raw = points.copy()
                    points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                    # Set initial pointcloud to target points
                    self.reg.set_input_target(points)
                    
                    num_trackable_points = trackable_filter.shape[0]
                    input_filter = np.zeros(points.shape[0], dtype=np.int32)
                    input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                    
                    self.reg.set_target_filter(num_trackable_points, input_filter)
                    self.reg.calculate_target_covariance_with_filter()

                    rots = self.reg.get_target_rotationsq()
                    scales = self.reg.get_target_scales()
                    rots = np.reshape(rots, (-1,4))
                    scales = np.reshape(scales, (-1,3))
                    
                    if self.ifma == 1:
                        H = round(current_image.shape[0] / 6 / 14) * 14
                        W = round(current_image.shape[1] / 6 / 14) * 14
                        if H < 224:
                            H = 224
                        if W < 140:
                            W = 140
                        descriptor = self.get_descriptor(current_image,(H,W))
                        #The image size should be times of 14 in both H and W, we resize the image to improve efficiency
                        self.keyframe_descriptors.append(descriptor)
                        
                        tmpInfo = [points_raw, num_trackable_points, input_filter, agentIndex, ii] 
                        #keyframe info: [pointcloud, num_trackable_points, filter, agentIndex, video_idx]
                        self.keyframe_info.append(tmpInfo)
                    
                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter), torch.tensor(agentIndex * self.max_frame_number))
                    
                    # Add first keyframe
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = agentIndex * self.max_frame_number
                    self.shared_cam.agent_idx[0] = agentIndex
                    
                    self.is_tracking_keyframe_shared[0] = 1
                    
                    while self.demo[0]:
                        time.sleep(1e-15)
                        self.total_start_time = time.time()
                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))
                else:
                    self.reg.set_input_source(points)
                    num_trackable_points = trackable_filter.shape[0]
                    input_filter = np.zeros(points.shape[0], dtype=np.int32)
                    input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                    self.reg.set_source_filter(num_trackable_points, input_filter)
                    
                    initial_pose = self.poses[agentIndex][-1]
                    current_pose = self.reg.align(initial_pose)
                    
                    #Here current_pose is the camera's pose
                    self.poses[agentIndex].append(current_pose)
                    if self.ifma == 1:
                        self.initial_estimates.insert(agentIndex * self.max_frame_number + ii, self.rawPose2GTSAMPose(current_pose))
                        rel_pose =self.rawPose2GTSAMPose(initial_pose).between(self.rawPose2GTSAMPose(current_pose))
                        self.graph.add(gtsam.BetweenFactorPose3(agentIndex * self.max_frame_number + ii - 1, agentIndex * self.max_frame_number + ii, rel_pose, self.prior_noise))
                                
                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(
                            "cam/current",
                            rr.Transform3D(translation=self.poses[agentIndex][-1][:3,3],
                                        rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[agentIndex][-1][:3,:3])).as_quat()))
                        )
                        rr.log(
                            "cam/current",
                            rr.Pinhole(
                                resolution=[self.W, self.H],
                                image_from_camera=self.cam_intrinsic,
                                camera_xyz=rr.ViewCoordinates.RDF,
                            )
                        )
                        rr.log(
                            "cam/current",
                            rr.Image(current_image)
                        )

                    # transform current points from camera coordinate to world frame
                    points_raw = points.copy()
                    # Use only trackable points when tracking
                    target_corres, distances = self.reg.get_source_correspondence() # get associated points source points
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))
                    
                    #Loop Closure and Global Optimization
                    if ii % self.lc_freq == 0 and self.ifma == 1: # Every X frames try loop closure
                        highestScore = 0
                        bestIndex = -1
                        best_other = -1
                        highestOther = 0
                        H = round(current_image.shape[0] / 6 / 14) * 14
                        W = round(current_image.shape[1] / 6 / 14) * 14
                        if H < 224:
                            H = 224
                        if W < 140:
                            W = 140
                        descriptor = self.get_descriptor(current_image,(H,W))
                        for index in range(len(self.keyframe_info)):
                            simScore = self.calculate_similarity(descriptor, self.keyframe_descriptors[index])
                            if simScore > self.similarity_threshold and simScore > highestScore:
                                highestScore = simScore
                                bestIndex = index
                            if simScore > self.similarity_threshold_inter and self.keyframe_info[index][3] != agentIndex:
                                if simScore > highestOther:
                                    highestOther = simScore
                                    best_other = index
                        if best_other != -1:
                            bestIndex = best_other
                        #if there exists loop closure, we will adjust the trajectory based on the highest similarity score
                        if bestIndex != -1:
                            # print(f'Loop Closure Detected, Agent{agentIndex}_Frame{ii} and Agent{self.keyframe_info[bestIndex][3]}_Frame{self.keyframe_info[bestIndex][4]} with Score: {highestScore}')
                            self.loop_detected += 1
                            rel_FGICP = pygicp.FastGICP()
                            #keyframe info: [pointcloud in world frame, num_trackable_points, filter, agentIndex, video_idx, pose in world frame]
                            # Set source (points2) and filter the current frame point cloud
                            rel_FGICP.set_input_source(points_raw)
                            rel_FGICP.set_source_filter(num_trackable_points, input_filter)

                            # Set target (points1) and filter the history keyframe point cloud
                            target_raw = self.keyframe_info[bestIndex][0]
                            target_pose = self.poses[self.keyframe_info[bestIndex][3]][self.keyframe_info[bestIndex][4]]
                            inv_target = np.linalg.inv(target_pose) #current_pose is inversed for convenient pointcloud transform
                            T_tmp = inv_target[:3,3]
                            R_tmp = inv_target[:3,:3].transpose()
                            target_world = np.matmul(R_tmp, target_raw.transpose()).transpose() - np.matmul(R_tmp, T_tmp)
                            rel_FGICP.set_input_target(target_world)
                            rel_FGICP.set_target_filter(self.keyframe_info[bestIndex][1], self.keyframe_info[bestIndex][2])
                            
                            # Set maximum correspondence distance
                            rel_FGICP.set_max_correspondence_distance(self.max_correspondence_distance / 2)

                            transformation_prop = rel_FGICP.align(current_pose) #The proposed world pose of current frame by loop closure
                            rel_mat = self.rawPose2GTSAMPose(target_pose).between(self.rawPose2GTSAMPose(transformation_prop))
                            
                            # #Use GT relative pose in loop closure to prove that the accuracy limitation comes from relative pose estimation                       
                            self.graph.add(gtsam.BetweenFactorPose3(self.keyframe_info[bestIndex][3] * self.max_frame_number + self.keyframe_info[bestIndex][4], agentIndex * self.max_frame_number + ii, rel_mat,self.closure_noise))
                            if self.initial_estimates.exists(self.keyframe_info[bestIndex][3] * self.max_frame_number + self.keyframe_info[bestIndex][4]):
                                pass
                            else:
                                self.initial_estimates.insert(self.keyframe_info[bestIndex][3] * self.max_frame_number + self.keyframe_info[bestIndex][4], self.rawPose2GTSAMPose(target_pose))
                            parameters = gtsam.LevenbergMarquardtParams()
                            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, parameters)
                            # optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimates)
                            self.finalOptResult = optimizer.optimize()
        
                            
                            self.initial_estimates = self.finalOptResult
                            current_pose = self.GTSAMPose2RawPose(self.finalOptResult.atPose3(agentIndex * self.max_frame_number + ii)) #T_avg is optimized world pose of current frame

                            for ai in range(agentIndex + 1):
                                if ai == agentIndex:
                                    iter_num = ii
                                else:
                                    iter_num = self.trajmanager[ai].gt_poses.shape[0]
                                for key in range(iter_num):
                                    self.poses[ai][key] = self.GTSAMPose2RawPose(self.finalOptResult.atPose3(ai * self.max_frame_number + key))
                                    
                        
                        #Remapping the mapping keyframe based on optmized traj
                        
                        if bestIndex != -1:
                            for ai in range(agentIndex + 1):
                                for idx in range(len(self.mapping_keyframe_info[ai])): #self.mapping_keyframe_info[agentIndex] stores the frame_idx of keyframe
                                    #metadata: [points_raw,colors,z_values,trackable_filter,current_image,depth_image,camera_idx,T,R,rots,scales]
                                    metadata = self.mapping_keyframe_metadata[ai][idx].copy()
                                    tmpPose = self.poses[ai][self.mapping_keyframe_info[ai][idx]]
                                    tmpPose_inv = np.linalg.inv(tmpPose)
                                    T_new = tmpPose_inv[:3,3]
                                    R_new = tmpPose_inv[:3,:3].transpose()
                                    if np.linalg.norm(metadata[7] - T_new) >= 0.005 or self.rotation_matrix_difference(R_new, metadata[8]) > 5: #if pose estimation difference > 5mm or rotation difference > 5 execute gaussian update
                                        while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                                            time.sleep(1e-15)
                                        R_d_tmp = Rotation.from_matrix(R_new)    # from camera R
                                        R_d_q_tmp = R_d_tmp.as_quat()            # xyzw
                                        rots_tmp = self.quaternion_multiply(R_d_q_tmp, metadata[9])

                                        points_tmp = np.matmul(R_new, metadata[0].transpose()).transpose() - np.matmul(R_new, T_new)
                
                                        self.shared_new_gaussians.input_values(torch.tensor(points_tmp), torch.tensor(metadata[1]), 
                                                        torch.tensor(rots_tmp), torch.tensor(metadata[10]), 
                                                        torch.tensor(metadata[2]), torch.tensor(trackable_filter), torch.tensor(agentIndex * self.max_frame_number + ii))
                                       # Add new keyframe
                                        self.shared_cam.setup_cam(R_new, T_new, metadata[4], metadata[5])
                                        self.shared_cam.cam_idx[0] = metadata[6]
                                        self.shared_cam.agent_idx[0] = ai
                                        self.is_mapping_keyframe_shared[0] = 1
                                        #Update saved mapping keyframe info
                                        self.mapping_keyframe_metadata[ai][idx][7] = T_new
                                        self.mapping_keyframe_metadata[ai][idx][8] = R_new
                                    
                    
                    current_pose = np.linalg.inv(current_pose) #current_pose is inversed for convenient pointcloud transform
                    T = current_pose[:3,3]
                    R = current_pose[:3,:3].transpose()
                    
                    points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                    #proposed points in world coordinate
                
                    # Keyframe selection #
                    # Tracking keyframe
                    len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th
                    
                    if  (self.iteration_images >= self.num_images-1 \
                        or len_corres/distances.shape[0] < self.keyframe_th):
                        if_tracking_keyframe = True
                        self.from_last_tracking_keyframe = 0
                    else:
                        if_tracking_keyframe = False
                        self.from_last_tracking_keyframe += 1
                    
                    # Mapping keyframe
                    if (self.from_last_tracking_keyframe) % (self.keyframe_freq) == 0:
                        if_mapping_keyframe = True
                    else:
                        if_mapping_keyframe = False
                    
                    
                    if if_tracking_keyframe:
                        
                        while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                            time.sleep(1e-15)
                        
                        if self.ifma == 1:
                            ##First we should save descriptors
                            H = round(current_image.shape[0] / 6 / 14) * 14
                            W = round(current_image.shape[1] / 6 / 14) * 14
                            if H < 224:
                                H = 224
                            if W < 140:
                                W = 140
                            descriptor = self.get_descriptor(current_image,(H,W))
                            #The image size should be times of 14 in both H and W, we resize the image to improve efficiency
                            self.keyframe_descriptors.append(descriptor)
                        
                            tmpInfo = [points_raw, num_trackable_points, input_filter, agentIndex, ii] 
                            #keyframe info: [pointcloud, num_trackable_points, filter, agentIndex, video_idx]
                            self.keyframe_info.append(tmpInfo)
                        # Erase overlapped points from current pointcloud before adding to map gaussian #
                        # Using filter
                        R_d = Rotation.from_matrix(R)    # from camera R
                        R_d_q = R_d.as_quat()            # xyzw
                        rots = self.quaternion_multiply(R_d_q, rots)
                        not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                        trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]
                        
                        # Add new gaussians
                        self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter), torch.tensor(agentIndex * self.max_frame_number + ii))
                        # Add new keyframe
                        depth_image = depth_image.astype(np.float32)/self.depth_scale
                        self.shared_cam.setup_cam(R, T, current_image, depth_image)
                        self.shared_cam.cam_idx[0] = agentIndex * self.max_frame_number + ii
                        self.shared_cam.agent_idx[0] = agentIndex
       
                        self.is_tracking_keyframe_shared[0] = 1
                        
                        # Get new target point
                        while not self.target_gaussians_ready[0]:
                            time.sleep(1e-15)
                        target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                        self.reg.set_input_target(target_points)
                        self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                        self.target_gaussians_ready[0] = 0
                          
                        if self.rerun_viewer:
                            # rr.set_time_sequence("step", self.iteration_images)
                            rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                            rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

                    elif if_mapping_keyframe:    
                        while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                            time.sleep(1e-15)
                       
                        rots_raw = rots.copy()
                        R_d = Rotation.from_matrix(R)    # from camera R
                        R_d_q = R_d.as_quat()            # xyzw
                        rots = self.quaternion_multiply(R_d_q, rots)
                        
                        # Add new keyframe
                        depth_image = depth_image.astype(np.float32)/self.depth_scale
                        self.shared_cam.setup_cam(R, T, current_image, depth_image)
                        self.shared_cam.cam_idx[0] = agentIndex * self.max_frame_number + ii
                        self.shared_cam.agent_idx[0] = agentIndex
                        if self.ifma == 1:
                            self.mapping_keyframe_metadata[agentIndex].append([points_raw,colors,z_values,trackable_filter,current_image,depth_image,agentIndex * self.max_frame_number + ii,T,R,rots_raw,scales])     
                        
                        not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                        
                        # Add new gaussians
                        self.mapping_keyframe_info[agentIndex].append(ii)
                        self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter), torch.tensor(agentIndex * self.max_frame_number + ii))
                        self.is_mapping_keyframe_shared[0] = 1
                
                pbar.update(1)
                
                    
                self.iteration_images += 1
            
            # Tracking of one agent end
            pbar.close()
            if self.ifma == 1:
                parameters = gtsam.LevenbergMarquardtParams()
                optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, parameters)
                self.finalOptResult = optimizer.optimize()
            print(f"System FPS of Agent{agentIndex}: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        
        print(f"total tracking keyframe: {len(self.keyframe_info)}; total loop closure: {self.loop_detected}")
        for agentIndex in range(len(self.final_pose)):
            if self.ifma == 1:
                for key in range(self.trajmanager[agentIndex].gt_poses.shape[0]):
                    self.poses[agentIndex][key] = self.GTSAMPose2RawPose(self.finalOptResult.atPose3(agentIndex * self.max_frame_number + key))

            self.final_pose[agentIndex][:,:,:] = torch.tensor(self.poses[agentIndex]).float()
            ate, est_aligned = self.evaluate_ate(self.trajmanager[agentIndex].gt_poses, self.poses[agentIndex])
            print(f"ATE RMSE of Agent{agentIndex}: {ate*100.:.2f}")
            pic_path = os.path.join(self.output_path,f'Agent{agentIndex}trajVis.jpg')
            self.trajmanager[agentIndex].save_traj(est_aligned.T, pic_path)
            tmp = torch.from_numpy(est_aligned)
            torch.save(tmp, os.path.join(self.output_path,f'Agent{agentIndex}_est.pt'))
        #Finish All tracking
        self.end_of_dataset[0] = 1


    
    def get_images(self, images_folder, agentIndex):
        rgb_images = []
        depth_images = []
        if self.trajmanager[agentIndex].which_dataset == "replica" or self.trajmanager[agentIndex].which_dataset == "real":
            image_files = os.listdir(images_folder)
            # #for perturbed test
            # image_files = [file for file in os.listdir(images_folder)
            #    if file.lower().startswith('p')]
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files): 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                # depth_image_name = f"depth{image_name[7:]}" # for perturb test
                rgb_image = cv2.imread(f"{self.dataset_path[agentIndex]}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path[agentIndex]}/depth_images/{depth_image_name}.png"))
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1],depth_image.shape[0]))
                # #if add perturb in RGB
                # rgb_image = rgb_image.astype(float) + np.random.normal(0, 10, rgb_image.shape)
                
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
                # rgb_images.append(rgb_image.clip(0,255).astype(np.uint8))
            return rgb_images, depth_images
        elif self.trajmanager[agentIndex].which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager[agentIndex].color_paths))):
                rgb_image = cv2.imread(self.trajmanager[agentIndex].color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager[agentIndex].depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img): 
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        # colors += torch.randn(colors.shape) * 0.04
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        z_values += torch.randn(z_values.shape) * self.ifnoise
        z_values[z_values > self.depth_trunc] = 0
        z_values[z_values < 0] = 0
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]      
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def eliminate_overlapped2(self, distances, threshold):
        new_p_indices = np.where(distances>threshold)    # 5e-5
        new_p_indices = np.array(new_p_indices).flatten()
        return new_p_indices
    
    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        # print(rot)
        # print(trans)
        model_aligned = rot * model + trans
        alignment_error = model_aligned - data
        # alignment_error = np.matrix(model) - np.matrix(data)

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]
        

        return model_aligned, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        est_aligned, trans_error = self.align(est_traj_pts, gt_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error, est_aligned
