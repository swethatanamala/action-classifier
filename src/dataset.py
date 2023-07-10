import cv2
import os
import random
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from . import transforms as tsfms

class CookingDataset(Dataset):
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None):
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/*/*/*.jpg"))
        self.mode = mode
        random.seed(42)
        random.shuffle(self.all_images)
        self.train_val_dict = self.get_split()
        if data_limit:
            for key in ["images", "targets"]:
                self.train_val_dict[mode][key] = self.train_val_dict[mode][key][:data_limit]
        assert len(self.train_val_dict[mode]["images"]) == len(self.train_val_dict[mode]["targets"]), \
            "images and targets length should match"
        self.images, self.targets = self.train_val_dict[mode]["images"], self.train_val_dict[mode]["targets"]
        print(mode, len(self.images))
        self.transforms = transforms


    def get_split(self):
        names = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(x))) 
                                 for x in self.all_images])))
        train_len = int(len(names) * 0.65) + 1
        val_len = int(len(names) * 0.9)
        train_val_dict = {"train": 
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[:train_len] if name in filepath]},
                         "val":
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[train_len:val_len] if name in filepath]},
                         "test":
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[val_len:] if name in filepath]}
                        }
        for mode in ["train", "val", "test"]:
            gts = []
            for filepath in train_val_dict[mode]["images"]:
                sub_folder = os.path.basename(os.path.dirname(filepath))
                if "ingredients" in sub_folder:
                    gts.append(1)
                elif "stir" in sub_folder:
                    gts.append(2)
                else:
                    gts.append(0)
            train_val_dict[mode]["targets"] = gts
        return train_val_dict

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        target = self.targets[index]
        img = cv2.imread(img_path) / 255
        if self.transforms:
            return self.transforms[self.mode](img), target
        else:
            return img, target
        

def get_dataloaders(args, folder):
    transforms = {
        "train": tsfms.Compose([
            #tsfms.RandomBrightnessJitter(1),
            #tsfms.RandomSaturationJitter(1),
            #tsfms.RandomContrastJitter(1),
            tsfms.RandomIntensityJitter(0.9, 0.9, 0.9),
            tsfms.RandomNoise(0.2),
            #tsfms.RandomSizedCrop(512, frac_range=[0.08, 1]),
            tsfms.RandomRotate(20),
            tsfms.RandomHorizontalFlip(),
            tsfms.Resize((512, 512)),
            tsfms.Clip(),
            tsfms.ToTensor(),
        ]),
        "val": tsfms.Compose([
            tsfms.Resize((512, 512)),
            tsfms.Clip(),
            tsfms.ToTensor()
        ])
    }
    train_limit = None
    val_limit = None
    if args.data_limit:
        train_limit = int(args.data_limit * 0.7)
        val_limit = int(args.data_limit * 0.3)
    train_dataset = CookingDataset(folder, data_limit=train_limit, transforms=transforms)
    val_dataset = CookingDataset(folder, mode='val', data_limit=val_limit, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # Create your train data loader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)  # Create your validation data loader
    
    return {"train": train_loader,
            "val": val_loader}

