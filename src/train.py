import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn import metrics


def run_epoch(model, data_loaders, mode, epoch, num_epochs, criterion, optimizer, scheduler, device, writer):
    running_loss = 0
    predicted_labels_list = []
    ground_truth_labels_list = []
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    for images, targets in tqdm(data_loaders[mode]):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(images)
            #print("outputs", outputs)
            #print("targets", targets)
            loss = criterion(outputs, targets)
            if mode == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            output_labels = torch.argmax(outputs, dim=1)
            predicted_labels_list.extend(output_labels.cpu().numpy())
            ground_truth_labels_list.extend(targets.cpu().numpy())
    if mode == 'train':
        scheduler.step()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print('learning rate at ', epoch + 1, lr)
            writer.add_scalar('lr', lr, epoch + 1)
    epoch_loss = running_loss / len(data_loaders['train'])
    writer.add_scalar(f"Loss/{mode}", epoch_loss, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Loss: {epoch_loss:.4f}")
    acc = metrics.accuracy_score(ground_truth_labels_list, predicted_labels_list)
    writer.add_scalar(f"Accuracy/{mode}", acc, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Accuracy: {acc:.4f}")
    print(metrics.confusion_matrix(ground_truth_labels_list, predicted_labels_list))
    return acc
