import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from robustness import brightness,contrast,spatter,zoom_blur,motion_blur,defocus_blur,gaussian_noise,shot_noise,impulse_noise,speckle_noise,gaussian_blur,glass_blur,fog,frost,snow,jpeg_compression,pixelate,gaussian_noise_strong
from robustness_depth import depth_add_random_mask, depth_add_fixed_mask, depth_add_edge_erosion, depth_add_gaussian_noise, depth_range
from PIL import Image
from tqdm import tqdm
import random
import imageio
import robustness_exp_utils.datautils as datautils
        
class Perturb_Image():
    def __init__(self, args):
        super().__init__()
        self.dataset_path = args.dataset_path
        self.perturb_type = int(args.perturb_type)
        self.perturb_severity = int(args.perturb_severity)
        self.perturb_dynamic = int(args.perturb_dynamic)
        self.frame_downsample = int(args.frame_downsample)

        self.output_path = f"/media/hdd/xiaohao/data/cpslam_{self.perturb_type}_{self.perturb_severity}_{self.perturb_dynamic}/"
        os.makedirs(self.output_path, exist_ok=True)
        
    def add_perturb(self, src, perturb_type, perturb_severity, isColor):
        if isColor:
            if perturb_type == 0:
                res = brightness(src,perturb_severity)
            elif perturb_type == 1:
                res = contrast(src,perturb_severity)
            elif perturb_type == 2:
                res = spatter(src,perturb_severity)
            elif perturb_type == 3:
                res = zoom_blur(src,perturb_severity)
            elif perturb_type == 4:
                res = motion_blur(src,perturb_severity)
            elif perturb_type == 5:
                res = defocus_blur(src,perturb_severity)
            elif perturb_type == 6:
                res = gaussian_noise(src,perturb_severity)
            elif perturb_type == 7:
                res = shot_noise(src,perturb_severity)
            elif perturb_type == 8:
                res = impulse_noise(src,perturb_severity)
            elif perturb_type == 9:
                res = speckle_noise(src,perturb_severity)
            elif perturb_type == 10:
                res = gaussian_blur(src,perturb_severity)
            elif perturb_type == 11:
                res = glass_blur(src,perturb_severity)
            elif perturb_type == 12:
                res = fog(src,perturb_severity)
            elif perturb_type == 13:
                res = frost(src,perturb_severity)
            elif perturb_type == 14:
                res = snow(src,perturb_severity)
            elif perturb_type == 15:
                res = jpeg_compression(src,perturb_severity)
            elif perturb_type == 16:
                res = pixelate(src,perturb_severity)
            elif perturb_type == 17 or perturb_type >= 20:
                res = src
        else:
            
            if perturb_type == 20:
                res = depth_add_gaussian_noise(src, perturb_severity)
            elif perturb_type == 21:
                res = depth_add_edge_erosion(src, perturb_severity)
            elif perturb_type == 22:
                res = depth_add_random_mask(src, perturb_severity)
            elif perturb_type == 23:
                res = depth_add_fixed_mask(src, perturb_severity)
            elif perturb_type >= 25 or perturb_type < 20:
                res = src

            if perturb_type == 24:
                res = depth_range(src,perturb_severity)
        return res
    
    
    def process(self):
        scenes = ['Apart-0/apart_0_part1','Apart-0/apart_0_part2','Apart-1/apart_1_part1','Apart-1/apart_1_part2','Apart-2/apart_2_part1','Apart-2/apart_2_part2','Office-0/office_0_part1','Office-0/office_0_part2']
        for scene in scenes:
            images_folder = f"{self.dataset_path}/{scene}/images"
            
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            index = 0
            total_num = len(image_files)
            os.makedirs(f"{self.output_path}{scene}/images", exist_ok=True)
            os.makedirs(f"{self.output_path}{scene}/depth_images", exist_ok=True)
            for key in tqdm(image_files):
                if self.frame_downsample != 0 and index % self.frame_downsample!=0:
                    index += 1
                    continue 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                
                rgb_image = cv2.imread(f"{self.dataset_path}/{scene}/images/{image_name}.jpg")
                depth = np.array(o3d.io.read_image(f"{self.dataset_path}/{scene}/depth_images/{depth_image_name}.png"))
                

                color = cv2.resize(rgb_image,(depth.shape[1], depth.shape[0]))
                cv2.imwrite(f"{self.output_path}{scene}/images/frame{str(int(index/self.frame_downsample)).zfill(6)}.jpg",color) #original image
                
                if self.perturb_dynamic > 0:
                    rnd_severity = random.choice([0, 1, 2, 3, 4, 5])
                    if rnd_severity > 0:
                        perturb_type = self.perturb_type
                        perturb_severity = rnd_severity
                    else:
                        perturb_type = 17 # no perturbation
                        perturb_severity = self.perturb_severity
                    print("dyn_perturb: "+ str(perturb_type) + "_" + str(perturb_severity))
                else:
                    perturb_type = self.perturb_type
                    perturb_severity = self.perturb_severity

                color = self.add_perturb(color,perturb_type,perturb_severity, True)
                color = np.array(color, dtype=np.uint8)
                cv2.imwrite(f"{self.output_path}{scene}/images/perturb{str(int(index/self.frame_downsample)).zfill(6)}.jpg",color) #perturb image

                depth = self.add_perturb(depth,perturb_type,perturb_severity, False)
                cv2.imwrite(f"{self.output_path}{scene}/depth_images/depth{str(int(index/self.frame_downsample)).zfill(6)}.png",depth) #perturb depth

                index += 1
            
            txt_input = f"{self.dataset_path}/{scene}/traj.txt"
            with open(txt_input, 'r') as infile:  #new traj if downsample frames
                lines = infile.readlines()
            
            extracted_lines = [lines[i] for i in range(0, len(lines), self.frame_downsample)]
            
            txt_output = f"{self.output_path}{scene}/traj.txt"
            with open(txt_output, 'w') as outfile:
                outfile.writelines(extracted_lines)

            print(scene + 'is done')

       

if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    # parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument('--perturb_type', type=int, default=17,help='')
    parser.add_argument('--perturb_severity', type=int, default=1,help='')
    parser.add_argument('--perturb_dynamic', type=int, default=0,help='')
    parser.add_argument('--frame_downsample', type=int, default=1,help='')
    args = parser.parse_args()

    perturbation = Perturb_Image(args)
    perturbation.process()