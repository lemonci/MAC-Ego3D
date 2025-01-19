# MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction
> Xiaohao Xu*, Feng Xue*, Shibo Zhao, Yike Pan, Sebastian Scherer, Xiaonan Huang

> University of Michigan, Ann Arbor & Carnegie Mellon University

> **The full code will be released in 2025 Feb.** Please stay tuned! :smiley:

[**:heart: Video Demo**](https://youtu.be/JOLQI_MNGAQ) [**:star: ArXiv Paper**](https://arxiv.org/abs/2412.09723)


## Pipeline: Multi-Agent Gaussian Consensus :raised_hands:

![image](https://github.com/user-attachments/assets/0a91e6ad-89a2-4eb4-95bd-4d7f2a3a4d3d)

**MAC-Ego3D** leverages parallel **Intra-Agent Gaussian Consensus** and periodic **Inter-Agent
Gaussian Consensus** to enable **real-time** pose tracking and **photorealistic** 3D reconstruction using a shared 3D Gaussian map representation.


---

## Environment Setup

To set up the required environment, follow these steps:

1. **Create and activate a new Conda environment:**

   ```bash
   conda create -n macego python==3.9
   conda activate macego
   ```

2. **Install necessary dependencies:**

   ```bash
   conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install nvidia/label/cuda-11.8.0::cuda-toolkit
   pip install -r requirements.txt
   ```

3. **Install PCL for `fast-gicp` submodule:**

   ```bash
   pip install pcl
   ```

4. **Install additional submodules:**

   ```bash
   conda activate macego
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/simple-knn
   ```

5. **Build and install the `fast_gicp` submodule:**

   ```bash
   cd submodules/fast_gicp
   mkdir build
   cd build
   cmake ..
   make
   cd ..
   python setup.py install --user
   ```

6. **Download model weights for the `salad` submodule:**

   Follow the link to download the weights:  
   [Loop Closure Detection Model Weights](https://drive.google.com/file/d/1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1/view)

   After downloading, place the weights in the `submodules/salad` directory.

---

## Datasets

### Replica (Multi-Agent Version from [this repo](https://huggingface.co/datasets/wssy37/CP-SLAM_dataset))

1. **Download the Replica dataset:**

   ```bash
   bash download_replica.sh
   ```

2. **Adjust directory structure:**

   Update the directory structure to match the following:

   **Original Structure:**
   ```bash
   Replica
      - {scene_name}
        - {scene_name_agent_id}
          - results (contain rgbd images)
              - frame000000.jpg
              - depth000000.png
              ...
          - traj.txt
   ```

   **Required Structure:**
   ```bash
   Replica
      - {scene_name}
        - {scene_name_agent_id}
           - images (contain rgb images)
               - frame000000.jpg
               ...
           - depth_images (contain depth images)
               - depth000000.png
               ...
           - traj.txt
   ```

### 7Scenes Dataset

1. **Download the 7Scenes dataset:**  
   [7Scenes Dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

2. **Configure the dataset:**  
   Some sequences in certain scenes may have missing frames. Make sure to remove any empty folders after processing.

   ```bash
   python 7Scene2ICP.py  # Modify the script to reflect the correct dataset and output paths
   ```

---


## Running the Code

### Multi-Agent Replica Dataset

To run all the experiments on the Multi-Agent Replica dataset:

```bash
bash multitest.sh
```

### 7Scenes Dataset

To run all the experiments on the 7Scenes dataset:

```bash
bash multitest_7scene.sh
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{xu2024macego3d,
  title={MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction},
  author={Xu, Xiaohao and Xue, Feng and Zhao, Shibo and Pan, Yike and Scherer, Sebastian and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2412.09723},
  year={2024}
}
```



## Contact

For questions or further inquiries, please report an issue or reach out to:  **Xiaohao Xu**  
Email: [xiaohaox[at]umich.edu](mailto:xiaohaox[at]umich.edu)
