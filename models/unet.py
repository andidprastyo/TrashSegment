import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import numpy as np

from segmentationModel import SegmentationModel

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetSegmenter(SegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = args.unet_model_path if hasattr(args, 'unet_model_path') else args.model_path
        
        # Set class names of interest
        self.target_class = 'floating_trash'
        self.barrier_class = 'barrier'
        
        # Input image resolution
        self.input_height = 512
        self.input_width = 512
        
        # Number of classes (background, floating_trash, barrier)
        self.n_classes = 3
        
        # Set per location where to remove out of system predictions
        # axis to look at -- above or below to remove
        self.remove_barrier = {
            '1': [0, 'above'],
            '2': [0, 'above'],
            '3': [0, 'below'],
            '4': [0, 'above'],
            '5': [0, 'below'],
            '6': [1, 'below']
        }
        
        # Class indices
        self.class_indices = {
            'background': 0,
            'floating_trash': 1,
            'barrier': 2
        }
    
    def preprocess(self):
        # Initialize the U-Net model
        self.model = UNet(n_channels=3, n_classes=self.n_classes)
        
        # Load pre-trained weights if available
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded U-Net model from {self.model_path}")
        except Exception as e:
            print(f"Could not load model weights: {e}")
            print("Using untrained model! Please train the model first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def predict(self, data):
        # Get data in correct format
        img = self.get_img_data(data)
        self.original_image_shape = img.shape[-2:]
        
        # Resize image to model input size
        input_img = resize(img, (self.input_height, self.input_width)).float()
        
        # Add batch dimension and move to device
        input_img = input_img.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_img)
            output = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        
        # Get predicted class for each pixel (argmax across channel dimension)
        predicted_masks = torch.argmax(output, dim=1)
        
        # Resize back to original resolution
        predicted_masks = F.interpolate(
            predicted_masks.unsqueeze(1).float(), 
            size=self.original_image_shape, 
            mode='nearest'
        ).squeeze(1).cpu()
        
        # Extract masks for trash and barrier classes
        trash_class_id = self.class_indices['floating_trash']
        barrier_id = self.class_indices['barrier']
        
        # If nothing is predicted, return zeros
        if torch.sum(predicted_masks == trash_class_id) == 0:
            return torch.zeros(self.original_image_shape).unsqueeze(0) > 0
        
        # Convert to binary masks for each class
        trash_mask = predicted_masks == trash_class_id
        barrier_mask = predicted_masks == barrier_id
        
        # If barrier is detected, apply the spatial filtering logic
        if torch.sum(barrier_mask) > 0:
            # Convert to list of individual trash masks
            connected_components = self._get_connected_components(trash_mask[0])
            trash_masks = []
            
            # Process barrier
            barrier_center = [np.average(indices) for indices in np.where(barrier_mask[0].numpy() >= 1)]
            
            # Filter trash components based on barrier position
            for component in connected_components:
                mask_center = [np.average(indices) for indices in np.where(component.numpy() >= 1)]
                
                # Apply location-specific filtering
                axis, above_below = self.remove_barrier[self.location]
                should_keep = True
                
                if above_below == 'below':
                    if barrier_center[axis] < mask_center[axis]:
                        should_keep = False
                else:  # 'above'
                    if barrier_center[axis] > mask_center[axis]:
                        should_keep = False
                
                if should_keep:
                    trash_masks.append(component)
            
            # If no trash masks remain after filtering, return empty
            if not trash_masks:
                return torch.zeros(self.original_image_shape).unsqueeze(0) > 0
            
            # Stack filtered trash masks
            return torch.stack(trash_masks)
        else:
            # If no barrier, return all trash masks as connected components
            connected_components = self._get_connected_components(trash_mask[0])
            
            # If no trash components found, return empty mask
            if not connected_components:
                return torch.zeros(self.original_image_shape).unsqueeze(0) > 0
                
            # Return all trash components
            return torch.stack(connected_components)
    
    def _get_connected_components(self, binary_mask):
        """Extract connected components from a binary mask"""
        # Use scipy's connected components analysis
        from scipy import ndimage
        
        labeled_mask, num_features = ndimage.label(binary_mask.numpy())
        
        # Convert each label to a separate binary mask
        components = []
        for i in range(1, num_features + 1):
            component = torch.from_numpy(labeled_mask == i)
            components.append(component)
        
        return components