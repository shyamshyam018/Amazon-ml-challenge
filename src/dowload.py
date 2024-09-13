import os
import pandas as pd
from src.utils import download_images

def main():
    # Load the dataset
    dataset_file_path = r'S:\Amazon-ML\dataset\train.csv'  # Update path as needed
    df = pd.read_csv(dataset_file_path)

    # Extract image links
    image_links = df['image_link'].dropna().tolist()

    # Set the folder where images will be saved
    download_folder = r'S:\Amazon-ML\downloaded_dataset'  # Update path as needed

    # Download the images
    download_images(image_links, download_folder)

if __name__ == '__main__':
    main()
