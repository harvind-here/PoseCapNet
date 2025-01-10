import os
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Create base directory
    base_dir = os.path.join(os.getcwd(), 'mini_coco')
    os.makedirs(base_dir, exist_ok=True)

    # Download validation images (smaller dataset)
    val_images_url = 'http://images.cocodataset.org/zips/val2017.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    print("Downloading validation images...")
    download_file(val_images_url, os.path.join(base_dir, 'val2017.zip'))
    
    print("Downloading annotations...")
    download_file(annotations_url, os.path.join(base_dir, 'annotations.zip'))

    # Extract files
    print("Extracting files...")
    for zip_file in ['val2017.zip', 'annotations.zip']:
        with zipfile.ZipFile(os.path.join(base_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        os.remove(os.path.join(base_dir, zip_file))

    print(f"Dataset downloaded to {base_dir}")
    print("Contains: validation images (5K images) and annotations")

if __name__ == "__main__":
    main()