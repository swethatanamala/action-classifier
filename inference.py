import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import get_model
from src import transforms as tsfms
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, data_folder, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/*.jpg"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        img_path = self.all_images[index]
        img = cv2.imread(img_path) / 255
        if self.transforms:
            return self.transforms(img), os.path.basename(img_path)[:-len('.png')]
        else:
            return img, os.path.basename(img_path)[:-len('.png')]

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def action_predictions(folder_path, args):
    model = get_model(args)
    results = {}
    transforms = tsfms.Compose([
                    tsfms.Resize((512, 512)),
                    tsfms.Clip(),
                    tsfms.ToTensor()
                ])
    testdataset = TestDataset(folder_path, transforms=transforms)
    testdataloader = DataLoader(testdataset, batch_size=2, shuffle=False)
    model.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  
    model.to(device)
    img = cv2.imread(testdataset.all_images[0])
    for images, names in tqdm(testdataloader):
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
        predictions = torch.argmax(output, dim=1)
        for i in range(predictions.shape[0]):
            results[names[i]] = predictions[i].cpu().item()
    return results
            

        


folder_path = "/home/users/swetha/projects/personal/hand-segmentation/data/test/One-Pot_Chicken_Fajita_Pasta"
testdataset = TestDataset(folder_path)
checkpoint_path = "checkpoints/resnet101_full_bs8_more_aug/checkpoint_overfit.pth"
args = {"resume": checkpoint_path,
        "model_name": "resnet101",
        "num_classes": 3}
overfit = ''
if 'overfit' in os.path.basename(checkpoint_path):
    overfit = 'overfit_'
args = AttributeDict(args)
model = get_model(args)
results = action_predictions(folder_path, args)
df = pd.DataFrame.from_dict(results, orient='index', columns=['prediction'])
df.to_csv(os.path.basename(os.path.dirname(checkpoint_path)) + f"{overfit}{os.path.basename(folder_path)}.csv")


