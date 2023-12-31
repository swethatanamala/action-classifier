import argparse
import datetime
import torch
import torch.nn as nn
import os
import torch.optim as optim
from src.dataset import get_dataloaders
from src.utils import save_checkpoint, get_model
from src.train import run_epoch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

date = str(datetime.datetime.now().date())
parser = argparse.ArgumentParser(description='HandSegmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint to be resumed')
parser.add_argument('--exp_name', type=str, default='exp_' + date,
                    help='Experiment prefix for checkpoint')
parser.add_argument('--data_limit', type=int, default=None,
                    help='Number if data to be limited')
parser.add_argument('--batch_size', type=int, default=4,
                    help="give the batchsize")
parser.add_argument('--model_name', type=str, default='resnet18',
                    help="give the model name that is required")
parser.add_argument('--num_classes', type=int, default=2,
                    help="number of output classes")

args = parser.parse_args()

def main(model, data_loaders, num_epochs, criterion, optimizer, scheduler, device, args, writer):
    best_dice = float('-Inf')
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.to(device)
        train_dice = run_epoch(model, data_loaders, "train", epoch, num_epochs, criterion, optimizer, scheduler, device, writer)
        val_dice = run_epoch(model, data_loaders, "val", epoch, num_epochs, criterion, optimizer, scheduler, device, writer)
        os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)
        model_save_path = f"checkpoints/{args.exp_name}/checkpoint_overfit.pth"
        best_save_path = f"checkpoints/{args.exp_name}/checkpoint_best_save.pth"
        save_checkpoint(epoch, model, optimizer, model_save_path)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            save_checkpoint(epoch, model, optimizer, best_save_path)
        print(f"Best val score is {str(best_dice)} at epoch {str(best_epoch)}")


model = get_model(args)

writer = SummaryWriter(f"logs/{args.exp_name}")
writer.add_text("Experiment Name", args.exp_name)
folder = "/cache/datanas1/swetha/youcook2/manual"
dataloaders = get_dataloaders(args, folder)
num_epochs = 100  # Specify the number of training epochs
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Choose your device
weight = torch.Tensor([2, 1])
weight = weight.to(device)
criterion = nn.CrossEntropyLoss()  # Define your loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Define your optimizer
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
main(model, dataloaders, num_epochs, criterion, optimizer, scheduler, device, args, writer)
writer.close()