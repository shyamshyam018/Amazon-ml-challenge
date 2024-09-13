import os
import pandas as pd
import requests
from tqdm import tqdm  # For showing progress bar

def download_images_from_csv(csv_file, download_folder):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Process each image link
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading Images"):
        image_link = row['image_link']
        image_id = index  

        # Define the path where the image will be saved
        image_filename = os.path.join(download_folder, f"{image_id}.jpg")

        try:
            # Download the image
            response = requests.get(image_link, stream=True)
            response.raise_for_status()  # Check for request errors
            
            # Write the image to the file
            with open(image_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
        except Exception as e:
            print(f"Failed to download image {image_id}: {e}")

    print("All images downloaded.")

# Define file paths
csv_file_path = r'S:\Amazon-ML\dataset\train.csv' # Update this with the actual path to train.csv
download_folder_path = r'S:\Amazon-ML\downloaded_dataset'  # Update this with the desired download folder path

# Call the function
download_images_from_csv(csv_file_path, download_folder_path)
