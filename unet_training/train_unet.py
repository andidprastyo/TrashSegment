import argparse
import sys
import os

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RiverSegmentationDataset

def train_unet(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Initialize the UNet model
    model = UNet(in_channels=3, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Load the dataset
    train_dataset = RiverSegmentationDataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images = batch['images'].to(device)
            masks = batch['gt_masks_per_class'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), args.unet_model_path)
    print(f'Model saved to {args.unet_model_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet for river segmentation.')
    
    # Dataset arguments
    parser.add_argument('--annotation_path', default='../data/annotations/all_annotations.json', help='Path to annotation file')
    parser.add_argument('--images_path', default='../data/images', help='Path to images folder')
    parser.add_argument('--location', default='1', help='Location ID for the dataset')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    
    # Model saving
    parser.add_argument('--unet_model_path', default='../models/unet.pth', help='Path to save the trained UNet model')
    
    # Add n_patches argument
    parser.add_argument('--n_patches', type=int, default=1, help='Number of patches to divide the image into')
    
    # Add timeseries argument
    parser.add_argument('--timeseries', default=None, help='Path to timeseries data')
    
    # Add model argument
    parser.add_argument('--model', default='UNet', help='Model type (e.g., UNet, YOLO)')
    
    # Add prompt_imgs argument
    parser.add_argument('--prompt_imgs', default=[0], nargs='+', type=int, help='List of prompt image indices')
    
    # Add yolo_test_path argument
    parser.add_argument('--yolo_test_path', default=None, help='Path to YOLO test images')
    
    args = parser.parse_args()
    train_unet(args)