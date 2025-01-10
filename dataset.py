import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np


class COCODataset(Dataset):
    def __init__(self, img_dir, caption_file, keypoint_file, transform=None):
        with open(caption_file, 'r') as f:
            caption_data=json.load(f)  
        with open(keypoint_file, 'r') as f:
            keypoint_data=json.load(f)

        self.img_dir=img_dir
        self.transform=transform or transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
        self.captions={ann['image_id']:ann['caption'] for ann in caption_data['annotations']}
        self.keypoints={ann['image_id']:ann['keypoints'] for ann in keypoint_data['annotations']}
        self.img_ids=list(set(self.captions.keys())&set(self.keypoints.keys()))

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id=self.img_ids[idx]
        img_path=os.path.join(self.img_dir, f"{img_id:012d}.jpg")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img=self.transform(img)

        caption=self.captions[img_id]
        keypoints=torch.tensor(self.keypoints[img_id])
        return img, caption, keypoints



        
        