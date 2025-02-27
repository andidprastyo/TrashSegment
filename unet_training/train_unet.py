import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet, UNetSegmenter

# Load your dataset with appropriate transforms
train_dataset = RiverSegmentationDataset(args, split='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize the model
model = UNet(n_channels=3, n_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        # Get images and labels
        images = batch['img'].to(device)
        # Assuming you have segmentation masks with class indices
        masks = batch['segmentation_masks'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pt')