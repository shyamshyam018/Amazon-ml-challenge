import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Image file name and path
        img_name = os.path.join(self.root_dir, f"{idx}.jpg")
        
        # Load image
        if not os.path.isfile(img_name):
            raise FileNotFoundError(f"Image file {img_name} not found.")
        
        try:
            image = Image.open(img_name).convert("RGB")
        except IOError:
            raise IOError(f"Error loading image: {img_name}")
        
        # Load corresponding label
        label = self.data_frame.iloc[idx]
        entity_name = label['entity_name']
        entity_value = label['entity_value']
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'entity_name': entity_name, 'entity_value': entity_value}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset
dataset = CustomDataset(csv_file=r'S:\Amazon-ML\dataset\train1.csv', root_dir=r'S:\Amazon-ML\downloaded_dataset', transform=transform)

# Iterate through the dataset and print details
for idx in range(len(dataset)):
    item = dataset[idx]
    img_name = f"{idx}.jpg"
    entity_name = item['entity_name']
    entity_value = item['entity_value']
    print(f"Image {img_name} - Entity: {entity_name}, Value: {entity_value}")
