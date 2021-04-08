import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils.utils import projectPoints, vector_to_heatmaps, IMAGE_SIZE


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))
        
        fn_K_matrix = os.path.join(data_dir, "training_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))
            
        fn_anno = os.path.join(data_dir, "training_xyz.json")
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))
            
        self.image_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, image_name))
        keypoints = projectPoints(self.anno[idx], self.K_matrix[idx])
        heatmaps = vector_to_heatmaps(keypoints)
        
        image = self.image_transform(image)
        keypoints = torch.from_numpy(keypoints/IMAGE_SIZE)
        heatmaps = torch.from_numpy(np.float32(heatmaps))
        return {"image": image, "keypoints": keypoints, "heatmaps": heatmaps}


