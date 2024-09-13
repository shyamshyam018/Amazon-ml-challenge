import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
import re  # Regular expressions for data cleaning

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_images=500):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_images (int): Maximum number of images to load.
        """
        self.data_frame = pd.read_csv(csv_file).iloc[:max_images]  # Load first max_images rows
        self.root_dir = root_dir
        self.transform = transform

        # Filter out rows where the image does not exist in the directory
        self.data_frame = self.data_frame[self.data_frame.index.map(
            lambda idx: os.path.isfile(os.path.join(root_dir, f"{idx}.jpg"))
        )]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx}.jpg")

        if not os.path.isfile(img_name):
            raise FileNotFoundError(f"Image file {img_name} not found.")

        try:
            image = Image.open(img_name).convert("RGB")
        except IOError:
            raise IOError(f"Error loading image: {img_name}")

        label = self.data_frame.iloc[idx]
        entity_value = self.clean_label(label['entity_value'])  # Clean the label
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'entity_value': entity_value}

    def clean_label(self, label):
        """
        Extract numeric value from a string with possible non-numeric characters.
        Args:
            label (str): The label containing the value.
        Returns:
            float: The numeric value extracted from the label.
        """
        # Extract numeric value from the string using regular expressions
        match = re.search(r"[\d.]+", label)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Could not extract numeric value from label: {label}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset
dataset = CustomDataset(csv_file=r'S:\Amazon-ML\dataset\train.csv', root_dir=r'S:\Amazon-ML\downloaded_dataset', transform=transform, max_images=500)

# Create the DataLoader
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None entries
    images = [item['image'] for item in batch]
    labels = [item['entity_value'] for item in batch]

    # Debugging: Check the labels
    print(f"Labels before conversion: {labels}")

    if len(images) == 0:
        return None  # Handle empty batches
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return {'image': images, 'entity_value': labels}

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model definition
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 256)  # Custom fully connected layer
        self.regression = nn.Linear(256, 1)  # Output layer for regression (one float value)

    def forward(self, x):
        x = self.resnet(x)
        x = self.regression(x)
        return x

# Model, criterion, optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
criterion = nn.MSELoss()  # Loss function for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            if batch is None:
                continue  # Skip empty batches

            images = batch['image'].to(device)
            labels = batch['entity_value'].to(device)  # Already converted to tensor
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Run training
train(model, dataloader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')  # Save model
print("Model saved successfully!")
